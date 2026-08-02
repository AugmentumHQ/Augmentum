/**
 * coder.js — Coder mode UI orchestrator.
 *
 * Conversation-first layout: AI agent conversation with integrated
 * terminal panel, file tree, and slide-in editor.
 * Agent communication via conversation input or // in terminal.
 */

import * as Terminal from './terminal.js';
import * as Editor from './cm-editor.js';
import * as CodeMind from './codemind.js';
import { currentCoder } from './coder-instance.js';
import { app, showToast, escapeHtml, extractErrorMessage } from './app.js';
import { CoderStream, stripToolCallJSON, readRunSeq } from './coder-stream.js';
import { mountReviewPanel } from './coder-review.js';
import { getModels } from './model-cache.js';
import { CoderConversation } from './coder-conversation.js';
import { supportsThinkingToggleForModel } from './settings.js';
import {
  handleStageEvent as _coderStageEvent,
  startStreamingTracker as _coderStartStreamTracker,
  recordStreamedDelta as _coderRecordDelta,
  notifyAllClear as _coderProgressClear,
} from './coder-progress.js';
import { scheduleAutosize } from './utils/textarea-autosize.js';
import { MissionPanel } from './mission-panel.js';
import {
  loadPreviewableExtensions,
  isPreviewable,
  buildPreviewUrl,
} from './coder-file-preview.js';
import {
  initCoderSearch,
  toggleCoderSearch,
  openCoderSearch,
  closeCoderSearch,
  isCoderSearchOpen,
} from './coder-search.js';

// Track bug-finder workspaces we've already auto-opened. Persisted in
// localStorage so the launcher modal only pops up on the very first
// entry per workspace, not on every page reload. The Set mirrors
// localStorage in-memory so we don't hit storage on every check.
const _BUG_FINDER_AUTO_OPENED_KEY = 'augmentum.coder.bugFinderAutoOpened';
const _bugFinderAutoOpened = (() => {
  try {
    const raw = localStorage.getItem(_BUG_FINDER_AUTO_OPENED_KEY);
    if (raw) {
      const arr = JSON.parse(raw);
      if (Array.isArray(arr)) return new Set(arr);
    }
  } catch (_) { /* corrupted or unavailable storage — start fresh */ }
  return new Set();
})();

function _persistBugFinderAutoOpened() {
  try {
    localStorage.setItem(
      _BUG_FINDER_AUTO_OPENED_KEY,
      JSON.stringify([..._bugFinderAutoOpened]),
    );
  } catch (_) { /* private browsing / quota — in-memory Set still works */ }
}

async function _maybeOpenBugFinderForWorkspace(ws) {
  if (!ws || ws.kind !== 'bug_finder') return;
  const wsId = ws.id || ws.workspace_id;
  if (!wsId) return;
  // Only auto-open BF on the very first entry into a given bug_finder
  // workspace. Page refreshes and workspace re-selects fall through to
  // the manual entry point (the BF tile + audit button) — auto-popping
  // the overlay on every coder reload was intrusive.
  if (_bugFinderAutoOpened.has(wsId)) return;
  _bugFinderAutoOpened.add(wsId);
  _persistBugFinderAutoOpened();
  try {
    const mod = await import('./bug-finder.js');
    mod.launchForWorkspace({
      workspaceId: wsId,
      primaryModel: app?.state?.currentModel || '',
      verifierModel: ws.bug_finder_verifier_model || '',
      openLauncher: true,
    });
  } catch (err) {
    console.debug('[Coder] bug-finder activate failed', err);
  }
}

// --- Coding agents (Claude Code / Codex) connect modal --------------------
// Whether a Claude token is connected — gates Claude as a selectable agent in
// the composer (never a run box we show/hide; the composer is always present).

async function _refreshClaudeAgentStatus() {
  const statusEl = document.getElementById('coder-claude-status');
  const disc = document.getElementById('coder-claude-disconnect-btn');
  if (!statusEl) return;
  try {
    const r = await fetch('/api/coder/external/claude/status');
    const d = await r.json();
    if (d && d.connected) {
      const kind = d.kind === 'subscription' ? 'subscription'
        : d.kind === 'api_key' ? 'API key' : 'connected';
      statusEl.textContent = `Connected · ${kind}`;
      statusEl.className = 'cca-status is-connected';
      if (disc) disc.style.display = '';
      currentCoder().claudeConnected = true;
    } else {
      statusEl.textContent = 'Not connected';
      statusEl.className = 'cca-status';
      if (disc) disc.style.display = 'none';
      currentCoder().claudeConnected = false;
    }
  } catch (err) {
    statusEl.textContent = 'Not connected';
    statusEl.className = 'cca-status';
    currentCoder().claudeConnected = false;
    console.debug('[Coder] claude status failed', err);
  }
  _refreshComposerAgents();  // connection state changes which agents are valid
}

// Drop the /workspace/ prefix — every path lives there, so it's just noise.
function _shortClaudePath(s) { return (s || '').replace(/^\/workspace\//, ''); }
// "mcp__augmentum__memory_search" → "augmentum.memory_search" for readability.
function _shortClaudeTool(name) {
  if (!name) return '';
  return name.startsWith('mcp__') ? name.slice(5).replace(/__/g, '.') : name;
}

// Format one normalized run event into a single output line (or null to skip).
function _claudeRunEventLine(e) {
  if (e.kind === 'status') return `${e.text || ''}`.trimEnd();
  if (e.kind === 'started') return `▶ ${e.text || 'started'}`;
  if (e.kind === 'thinking') return e.text ? `${e.text}` : null;
  if (e.kind === 'file_change') return `✎ ${_shortClaudePath(e.path || e.tool || '')}`.trimEnd();
  if (e.kind === 'command_exec') return `$ ${e.text || ''}`.trimEnd();
  if (e.kind === 'tool_call' || e.kind === 'mcp_call') {
    return `⚙ ${[_shortClaudeTool(e.tool), _shortClaudePath(e.text)].filter(Boolean).join(' ')}`.trimEnd();
  }
  if (e.kind === 'message' && e.text) return e.text;
  if (e.kind === 'completed') return `✓ ${e.text || 'done'}`;
  if (e.kind === 'failed') return `✗ ${e.text || 'failed'}`;
  return null;
}

// CSS class giving each event kind its own visual weight in the transcript.
function _claudeRunEventClass(kind) {
  const map = {
    status: 'cca-line--status', started: 'cca-line--status',
    thinking: 'cca-line--thinking', message: 'cca-line--msg',
    file_change: 'cca-line--file', command_exec: 'cca-line--cmd',
    tool_call: 'cca-line--tool', mcp_call: 'cca-line--tool',
    completed: 'cca-line--done', failed: 'cca-line--failed',
  };
  return map[kind] || 'cca-line--msg';
}

// Append a styled transcript line; returns the created element (or null).
function _appendClaudeLine(container, e) {
  const text = _claudeRunEventLine(e);
  if (!container || !text) return null;
  const atBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - 4;
  const div = document.createElement('div');
  div.className = `cca-line ${_claudeRunEventClass(e.kind)}`;
  div.textContent = text;
  container.appendChild(div);
  if (atBottom) container.scrollTop = container.scrollHeight;  // follow tail unless scrolled up
  return div;
}

// "3 turns · 12s · $0.04" — surfaced run metadata (omits zero/empty parts).
function _claudeRunMeta(run) {
  const parts = [];
  if (run.num_turns) parts.push(`${run.num_turns} turn${run.num_turns === 1 ? '' : 's'}`);
  if (run.duration_ms) {
    const s = Math.round(run.duration_ms / 1000);
    parts.push(s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`);
  }
  if (run.cost_usd) parts.push(`$${Number(run.cost_usd).toFixed(run.cost_usd < 1 ? 4 : 2)}`);
  return parts;
}

// The run the next "Run with Claude" will continue (Claude's native --resume),
// or '' for a fresh session. Set by "Continue" in the run history.
// The run currently streaming into the panel (so Stop knows what to cancel).

function _setClaudeResume(runId, label) {
  currentCoder().claudeResumeRunId = runId || '';
  const banner = document.getElementById('coder-claude-resume-banner');
  const lbl = document.getElementById('coder-claude-resume-label');
  if (banner) banner.classList.toggle('hidden', !currentCoder().claudeResumeRunId);
  if (lbl && currentCoder().claudeResumeRunId) lbl.textContent = label || 'Continuing a previous run.';
  if (currentCoder().claudeResumeRunId) document.getElementById('cca-task')?.focus();
}

function _claudeSetStatus(statusEl, text, busy) {
  if (!statusEl) return;
  statusEl.innerHTML = busy ? '<span class="cca-spinner"></span>' : '';
  statusEl.appendChild(document.createTextNode(text));
}

function _claudeDoneText(e) {
  if (e.ok) {
    const meta = _claudeRunMeta(e);
    const m = meta.length ? ` · ${meta.join(' · ')}` : '';
    return `Done${e.files_changed?.length ? ` · ${e.files_changed.length} file(s)` : ''}${m}`;
  }
  if (e.error) return e._incomplete ? e.error : `Failed: ${e.error}`;
  return 'Ended without completing.';
}

// Dispatch↔Stop are mutually exclusive; show one or the other.
function _showClaudeStop(show) {
  document.getElementById('cca-stop-btn')?.classList.toggle('hidden', !show);
  document.getElementById('cca-dispatch-btn')?.classList.toggle('hidden', !!show);
}

// Shared SSE reader for both starting a run and re-attaching to one.
async function _consumeClaudeStream(resp, { onRun, onLine, onDone }) {
  if (!resp.ok || !resp.body) {
    let msg = 'Run failed.';
    try { const d = await resp.json(); msg = d.error || msg; } catch (_e) { /* non-JSON */ }
    onDone?.({ ok: false, error: msg, _incomplete: true });
    return;
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  let sawDone = false;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop() || '';
    for (const raw of lines) {
      const line = raw.trim();
      if (!line.startsWith('data:')) continue;
      const jsonStr = line.slice(5).trim();
      if (!jsonStr || jsonStr === '[DONE]') continue;
      let e;
      try { e = JSON.parse(jsonStr); } catch (_err) { continue; }
      if (e.kind === 'run') { onRun?.(e); continue; }
      if (e.kind === 'done') { sawDone = true; onDone?.(e); continue; }
      onLine?.(e);
    }
  }
  if (!sawDone) onDone?.({ ok: false, error: 'Stream ended unexpectedly.', _incomplete: true });
}

// Stream a Claude in-container SDK run into the composer output pane. The one
// live streaming path (Augmentum + harness dispatch are async/fire-and-list).
async function _streamClaudeRun({ task, wsId, permission = 'auto', resumeRunId = '', model = '' }) {
  const taskEl = document.getElementById('cca-task');
  const statusEl = document.getElementById('cca-status');
  const outEl = document.getElementById('cca-output');
  const runBtn = document.getElementById('cca-dispatch-btn');
  _claudeSetStatus(statusEl, resumeRunId ? 'Continuing…' : 'Running…', true);
  if (outEl) { outEl.style.display = ''; outEl.textContent = ''; }
  if (runBtn) runBtn.disabled = true;
  _showClaudeStop(true);

  try {
    const resp = await fetch('/api/coder/external/claude/run/stream', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace_id: wsId, task, resume_run_id: resumeRunId, permission, model }),
    });
    await _consumeClaudeStream(resp, {
      onRun: (e) => { currentCoder().claudeActiveRunId = e.run_id || ''; },
      onLine: (e) => _appendClaudeLine(outEl, e),
      onDone: (e) => _claudeSetStatus(statusEl, _claudeDoneText(e), false),
    });
  } catch (err) {
    _claudeSetStatus(statusEl, 'Request failed.', false);
    console.debug('[Coder] run with claude failed', err);
  } finally {
    if (runBtn) runBtn.disabled = false;
    _showClaudeStop(false);
    currentCoder().claudeActiveRunId = '';
    _setClaudeResume('', '');            // consumed — back to fresh sessions
    if (taskEl) taskEl.value = '';
    _refreshUnifiedRuns();               // reflect the run we just finished
  }
}

// Slash-command / programmatic entry: run a task with Claude in the active
// workspace (the classic "/claude <task>" behaviour), independent of the
// composer's current selection.
async function _runWithClaude() {
  const statusEl = document.getElementById('cca-status');
  const task = (document.getElementById('cca-task')?.value || '').trim();
  if (!task) { _claudeSetStatus(statusEl, 'Type a task first.', false); return; }
  const wsId = getActiveWorkspaceId();
  if (!wsId) { _claudeSetStatus(statusEl, 'Open a workspace first.', false); return; }
  await _streamClaudeRun({ task, wsId, permission: 'auto', resumeRunId: currentCoder().claudeResumeRunId });
}

// Re-attach to a run already executing server-side (it survived a refresh /
// modal close) and stream it into the panel. Closing here doesn't stop the run.
async function _attachClaudeRun(runId) {
  const statusEl = document.getElementById('cca-status');
  const outEl = document.getElementById('cca-output');
  if (outEl) { outEl.style.display = ''; outEl.textContent = ''; }
  _claudeSetStatus(statusEl, 'Attaching…', true);
  currentCoder().claudeActiveRunId = runId;
  _showClaudeStop(true);
  try {
    const resp = await fetch(`/api/coder/external/claude/run/${encodeURIComponent(runId)}/stream`);
    await _consumeClaudeStream(resp, {
      onLine: (e) => _appendClaudeLine(outEl, e),
      onDone: (e) => _claudeSetStatus(statusEl, _claudeDoneText(e), false),
    });
  } catch (err) {
    _claudeSetStatus(statusEl, 'Attach failed.', false);
    console.debug('[Coder] attach failed', err);
  } finally {
    _showClaudeStop(false);
    currentCoder().claudeActiveRunId = '';
    _refreshUnifiedRuns();
  }
}

async function _stopClaudeRun() {
  if (!currentCoder().claudeActiveRunId) return;
  const statusEl = document.getElementById('cca-status');
  _claudeSetStatus(statusEl, 'Stopping…', true);
  try {
    await fetch(`/api/coder/external/claude/run/${encodeURIComponent(currentCoder().claudeActiveRunId)}/stop`,
      { method: 'POST' });
    // The stream will receive the cancelled 'done' frame and settle the UI.
  } catch (err) {
    console.debug('[Coder] stop failed', err);
  }
}

// --- Unified agent-run history (every engine, grouped by derived locus) -----
//
// Fed by GET /api/coder/agents/runs → { internal:[row], external:[row] }.
// One row shape everywhere: { id, agent, locus, goal, status, model, turns,
// tools, cost_usd, duration_ms, result, where, run_id, review_turn_id,
// session_id, source, updated_at }. `source` (claude_run | pi_run |
// coding_run) selects the windowed-detail renderer — every one already
// exists; the agent is a badge, not a different UI.

const _AGENT_LABEL = { augmentum: 'Augmentum', claude: 'Claude', pi: 'pi' };
const _LIVE_STATES = new Set(['working', 'queued']);

function _agentBadge(agent) {
  const key = String(agent || '').toLowerCase();
  const label = _AGENT_LABEL[key] || agent || '?';
  return `<span class="cca-agent cca-agent--${escapeHtml(key)}">${escapeHtml(label)}</span>`;
}

// Status pill — spinner while live, coloured pill otherwise. Reuses the
// existing .cca-badge variants so it matches the rest of the modal.
function _agentStatusPill(status) {
  const s = String(status || '').toLowerCase();
  if (s === 'done') return '<span class="cca-badge is-done">done</span>';
  if (s === 'failed') return '<span class="cca-badge is-failed">failed</span>';
  if (s === 'cancelled') return '<span class="cca-badge">cancelled</span>';
  if (s === 'queued') return '<span class="cca-badge is-running">queued</span>';
  if (s === 'detached') return '<span class="cca-badge">detached</span>';
  return '<span class="cca-badge is-running"><span class="cca-spinner"></span> working</span>';
}

// "2026-07-20 14:03 · 3 turns · 5 tools · 12s · $0.04" — omit empty parts.
function _unifiedRunMeta(row) {
  const parts = [];
  if (row.updated_at) parts.push(String(row.updated_at).replace('T', ' ').slice(0, 16));
  if (row.turns) parts.push(`${row.turns} turn${row.turns === 1 ? '' : 's'}`);
  if (row.tools) parts.push(`${row.tools} tool${row.tools === 1 ? '' : 's'}`);
  if (row.duration_ms) {
    const sec = Math.round(row.duration_ms / 1000);
    parts.push(sec >= 60 ? `${Math.floor(sec / 60)}m ${sec % 60}s` : `${sec}s`);
  }
  if (row.cost_usd) parts.push(`$${Number(row.cost_usd).toFixed(row.cost_usd < 1 ? 4 : 2)}`);
  // Every engine now reports the real model it ran (Augmentum local id, Claude/
  // pi their own). Empty until captured (e.g. an in-flight Claude run).
  if (row.model) parts.push(row.model);
  return parts;
}

function _unifiedRunRowHtml(row) {
  const meta = _unifiedRunMeta(row).map((m) => escapeHtml(m)).join('<span class="cca-dot">·</span>');
  const live = _LIVE_STATES.has(String(row.status || '').toLowerCase());
  const id = escapeHtml(row.id || '');
  return `<div class="cca-run-row cca-run-row--click" data-run-id="${id}" tabindex="0" role="button">
    <div class="cca-run-row-main">
      ${_agentBadge(row.agent)}
      ${_agentStatusPill(row.status)}
      <span class="cca-run-task" title="${escapeHtml(row.goal || '')}">${escapeHtml(row.goal || '(no task)')}</span>
      ${live ? '<span class="cca-live-dot" title="live"></span>' : ''}
    </div>
    ${meta ? `<div class="cca-run-meta">${meta}</div>` : ''}
    ${row.result ? `<div class="cca-run-outcome">${escapeHtml(row.result)}</div>` : ''}
  </div>`;
}

// Last fetched rows, keyed by id, so a row click recovers the full object
// (the windowed detail needs source/run_id/review_turn_id/session_id).

function _renderLocusList(listEl, rows, countId, emptyMsg) {
  if (!listEl) return;
  listEl.innerHTML = rows.length
    ? rows.map(_unifiedRunRowHtml).join('')
    : `<div class="cca-history-empty">${escapeHtml(emptyMsg)}</div>`;
  const countEl = document.getElementById(countId);
  if (countEl) countEl.textContent = rows.length ? String(rows.length) : '';
}

async function _refreshUnifiedRuns() {
  const internalEl = document.getElementById('cca-internal-list');
  const externalEl = document.getElementById('cca-external-list');
  if (!internalEl || !externalEl) return;
  const wsId = getActiveWorkspaceId();
  const qs = wsId ? `?workspace_id=${encodeURIComponent(wsId)}` : '';
  try {
    const r = await fetch(`/api/coder/agents/runs${qs}`);
    const d = await r.json();
    const internal = Array.isArray(d.internal) ? d.internal : [];
    const external = Array.isArray(d.external) ? d.external : [];
    currentCoder().unifiedRunsById = new Map();
    for (const row of [...internal, ...external]) if (row.id) currentCoder().unifiedRunsById.set(row.id, row);
    _renderLocusList(internalEl, internal, 'cca-internal-count', 'No internal runs yet.');
    _renderLocusList(externalEl, external, 'cca-external-count',
      'No external runs. Connect Claude Code or pi to work on your own machine.');
  } catch (err) {
    internalEl.innerHTML = '<div class="cca-history-empty">Could not load runs.</div>';
    console.debug('[Coder] unified runs load failed', err);
  }
}

// --- Windowed detail view (click-into) — "extend the terminal into here" -----


function _detachAgentDetailStream() {
  const s = currentCoder().agentDetailStream;
  currentCoder().agentDetailStream = null;
  if (s && typeof s.abort === 'function') { try { s.abort(); } catch { /* noop */ } }
}

function _closeAgentDetail() {
  _detachAgentDetailStream();
  const el = document.getElementById('coder-agents-detail');
  if (el) { el.classList.add('hidden'); el.innerHTML = ''; }
}

async function _openAgentDetail(row) {
  const el = document.getElementById('coder-agents-detail');
  if (!el || !row) return;
  _detachAgentDetailStream();
  el.classList.remove('hidden');
  el.innerHTML = `
    <div class="cca-detail-head">
      <button class="cca-detail-back" type="button" title="Close">←</button>
      ${_agentBadge(row.agent)}
      ${_agentStatusPill(row.status)}
      <span class="cca-detail-goal" title="${escapeHtml(row.goal || '')}">${escapeHtml(row.goal || '')}</span>
    </div>
    <div class="cca-detail-body" id="cca-detail-body"><div class="cca-history-empty">Loading…</div></div>`;
  el.querySelector('.cca-detail-back')?.addEventListener('click', _closeAgentDetail);
  el.scrollIntoView?.({ block: 'nearest' });
  const body = el.querySelector('#cca-detail-body');
  try {
    if (row.source === 'claude_run') await _renderClaudeDetail(row, body);
    else if (row.source === 'pi_run') await _renderPiDetail(row, body);
    else await _renderCodingDetail(row, body);
  } catch (err) {
    body.innerHTML = '<div class="cca-history-empty">Could not load this run.</div>';
    console.debug('[Coder] agent detail failed', err);
  }
}

// Claude — live attach for in-flight runs, full transcript otherwise. Reuses
// the same SSE reader + line renderer as the composer's own output pane.
async function _renderClaudeDetail(row, body) {
  const box = document.createElement('div');
  box.className = 'cca-transcript';
  body.innerHTML = '';
  body.appendChild(box);
  const runId = row.run_id || row.id;
  if (_LIVE_STATES.has(String(row.status).toLowerCase()) && runId) {
    const resp = await fetch(`/api/coder/external/claude/run/${encodeURIComponent(runId)}/stream`);
    await _consumeClaudeStream(resp, {
      onLine: (e) => _appendClaudeLine(box, e),
      onDone: (e) => _appendClaudeLine(box, { kind: e.ok ? 'completed' : 'failed', text: _claudeDoneText(e) }),
    });
    _refreshUnifiedRuns();
    return;
  }
  const r = await fetch(`/api/coder/external/claude/runs/${encodeURIComponent(runId)}`);
  const d = await r.json();
  const events = d.run?.events || [];
  for (const e of events) _appendClaudeLine(box, e);
  if (!box.childNodes.length) box.innerHTML = '<div class="cca-line cca-line--status">(no transcript)</div>';
  // Continue is first-class — resume Claude's native session from anywhere.
  if (row.session_id) {
    const bar = document.createElement('div');
    bar.className = 'cca-detail-actions';
    const btn = document.createElement('button');
    btn.className = 'btn small';
    btn.textContent = 'Continue this run';
    btn.addEventListener('click', () => {
      const g = row.goal || '';
      _setClaudeResume(runId, `Continuing: "${g.slice(0, 50)}${g.length > 50 ? '…' : ''}"`);
      // Continue is Claude-only; make sure the composer is set to Claude on the
      // run's workspace so Dispatch resumes rather than starting a fresh run.
      _selectComposerFor({ agent: 'claude', wsId: row.workspace_id || getActiveWorkspaceId() });
      document.getElementById('cca-task')?.scrollIntoView?.({ block: 'center' });
    });
    bar.appendChild(btn);
    body.appendChild(bar);
  }
}

// pi — host-pushed terminal mirror; full relayed transcript.
async function _renderPiDetail(row, body) {
  const r = await fetch(`/api/coder/external/pi/runs/${encodeURIComponent(row.run_id || row.id)}`);
  const d = await r.json();
  const events = d.events || [];
  const box = document.createElement('div');
  box.className = 'cca-transcript';
  for (const e of events) _appendClaudeLine(box, e);
  if (!box.childNodes.length) box.innerHTML = '<div class="cca-line cca-line--status">(no transcript)</div>';
  body.innerHTML = '';
  body.appendChild(box);
}

// Internal mission (coding_run) — done+review → the REAL coder review panel
// (unified diff, per-file Partial, Accept/Reject); active → live stage feed;
// else a fallback diff. Harness assignments run on the user's machine.
async function _renderCodingDetail(row, body) {
  const status = String(row.status || '').toLowerCase();
  const active = _LIVE_STATES.has(status);
  if (row.locus === 'external') {
    body.innerHTML = '<div class="cca-history-empty">This agent runs on your own machine. Its work and approvals reach you through notifications — reply from any device.</div>';
    return;
  }
  if (active && row.run_id) {
    body.innerHTML = '<ul class="cca-stage-feed"></ul>';
    const feed = body.querySelector('.cca-stage-feed');
    const push = (label, cls) => {
      if (!feed || !label) return;
      const li = document.createElement('li');
      li.className = `cca-stage cca-stage--${cls || 'progress'}`;
      li.textContent = label;
      feed.appendChild(li); feed.scrollTop = feed.scrollHeight;
    };
    push('Reattaching to the running agent…', 'start');
    const stream = new CoderStream({
      onStage: (p) => {
        const label = p.label || p.stage || p.message || '';
        if (label) push(label, p.type === 'complete' ? 'done' : 'progress');
        if (p.type === 'complete') _refreshUnifiedRuns();
      },
      onError: (e) => push(String(e || 'stream error'), 'error'),
    });
    currentCoder().agentDetailStream = stream;
    stream.attach({ runId: row.run_id }).catch(() => push('Could not attach to the run stream.', 'error'));
    return;
  }
  if (status === 'done' && row.review_turn_id) {
    const mounted = await mountReviewPanel(row.review_turn_id, body);
    if (mounted) return;
  }
  await _renderCodingFallbackDiff(row.id, body, active);
}

async function _renderCodingFallbackDiff(runId, body, active) {
  body.innerHTML = '<div class="cca-history-empty">Loading changes…</div>';
  let data;
  try {
    const resp = await fetch(`/api/coding/runs/${encodeURIComponent(runId)}/diff`);
    data = resp.ok ? await resp.json() : { error: `Fetch failed (${resp.status})` };
  } catch (err) { data = { error: String(err?.message || err) }; }
  if (data.error) { body.innerHTML = `<div class="cca-history-empty">${escapeHtml(data.error)}</div>`; return; }
  const untracked = Array.isArray(data.untracked) ? data.untracked : [];
  const patch = data.patch || '';
  body.innerHTML = `
    ${data.stat ? `<pre class="cca-diff-stat">${escapeHtml(data.stat)}</pre>` : ''}
    ${untracked.length ? `<div class="cca-untracked"><strong>New files:</strong> ${untracked.map((f) => `<code>${escapeHtml(f)}</code>`).join(' ')}</div>` : ''}
    ${patch
      ? `<pre class="cca-diff-patch">${_renderUnifiedPatch(patch)}</pre>${data.truncated ? '<div class="cca-history-empty">diff truncated</div>' : ''}`
      : `<div class="cca-history-empty">No changes yet${active ? ' — the agent is still working.' : '.'}</div>`}`;
}

// Minimal +/- line coloring (escape first, then wrap).
function _renderUnifiedPatch(patch) {
  return patch.split('\n').map((line) => {
    const esc = escapeHtml(line);
    if (line.startsWith('+') && !line.startsWith('+++')) return `<span class="diff-add">${esc}</span>`;
    if (line.startsWith('-') && !line.startsWith('---')) return `<span class="diff-del">${esc}</span>`;
    if (line.startsWith('@@')) return `<span class="diff-hunk">${esc}</span>`;
    return esc;
  }).join('\n');
}

// --- New-run composer: task + target + agent (locus derived from target) ----
//
// Target answers "where must the work happen?" — a specific Augmentum
// workspace (→ internal, sandboxed) or the user's own machine (→ external,
// relayed). Agent answers "who does it?" and the valid set falls out of the
// target: a workspace can be driven by the Augmentum local model or Claude
// in-container; "my machine" is assigned to a live bridge agent that has
// checked in. Internal/external is never a control — it's a consequence of
// the target (see docs/superpowers/specs/2026-07-20-coding-agents-surface).

let _composerWorkspaces = [];       // [{id, name}] from GET /api/coder/workspaces
let _composerExternalAgents = [];   // [{agent_id, title, ...}] from GET /api/harness/agents
let _composerProviders = {};        // {id: {label, dispatch, model_targetable, models, enabled}}

async function _loadComposerProviders() {
  try {
    const r = await fetch('/api/coder/agents/providers');
    const d = await r.json();
    _composerProviders = {};
    for (const p of (Array.isArray(d.providers) ? d.providers : [])) {
      if (p && p.id) _composerProviders[p.id] = p;
    }
  } catch (err) {
    _composerProviders = {};
    console.debug('[Coder] composer providers load failed', err);
  }
}

async function _loadComposerExternalAgents() {
  try {
    const r = await fetch('/api/harness/agents');
    const d = await r.json();
    _composerExternalAgents = Array.isArray(d.agents) ? d.agents : [];
  } catch (err) {
    _composerExternalAgents = [];
    console.debug('[Coder] composer external agents load failed', err);
  }
}

async function _loadComposerTargets() {
  const sel = document.getElementById('cca-target');
  if (!sel) return;
  const prev = sel.value;
  try {
    const r = await fetch('/api/coder/workspaces');
    const d = await r.json();
    _composerWorkspaces = Array.isArray(d.workspaces) ? d.workspaces : [];
  } catch (err) {
    _composerWorkspaces = [];
    console.debug('[Coder] composer targets load failed', err);
  }
  const wsOpts = _composerWorkspaces.map((w) => {
    const id = w.id || '';
    return `<option value="ws:${escapeHtml(id)}">${escapeHtml(w.name || id)}</option>`;
  }).join('');
  sel.innerHTML = wsOpts + '<option value="machine">My machine</option>';
  // Default to the active workspace (the context the user is already in — a
  // sensible default, not a hidden engine/model pick), else keep prior.
  const activeWs = getActiveWorkspaceId();
  const want = (prev && [...sel.options].some((o) => o.value === prev)) ? prev
    : (activeWs ? `ws:${activeWs}` : '');
  if (want) sel.value = want;
  await _refreshComposerAgents();
}

async function _refreshComposerAgents() {
  const sel = document.getElementById('cca-agent');
  if (!sel) return;
  const target = document.getElementById('cca-target')?.value || '';
  const prev = sel.value;
  // Lead with a placeholder so nothing is auto-selected on the user's behalf.
  let opts = '<option value="">Choose agent…</option>';
  if (target.startsWith('ws:')) {
    // A workspace can be driven by the Augmentum local model, or any enabled
    // in-container SDK engine (Claude now, Codex when its driver lands — it
    // appears here automatically once providers.py flips enabled=true).
    opts += '<option value="augmentum">Augmentum (local model)</option>';
    for (const p of Object.values(_composerProviders)) {
      if (p.dispatch !== 'stream') continue;
      if (p.id === 'claude' && !currentCoder().claudeConnected) {
        opts += `<option value="claude" disabled>${escapeHtml(p.label)} — connect above</option>`;
      } else {
        opts += `<option value="${escapeHtml(p.id)}">${escapeHtml(p.label)}</option>`;
      }
    }
  } else if (target === 'machine') {
    if (_composerExternalAgents.length) {
      opts += _composerExternalAgents.map((a) => {
        const id = a.agent_id || '';
        const name = a.title || a.harness || id || 'agent';
        return `<option value="harness:${escapeHtml(id)}">${escapeHtml(name)}</option>`;
      }).join('');
    } else {
      opts += '<option value="" disabled>No agents connected on your machine</option>';
    }
  }
  sel.innerHTML = opts;
  if (prev && [...sel.options].some((o) => o.value === prev && !o.disabled)) sel.value = prev;
  _onComposerAgentChange();
}

// Populate the Model control for the selected agent. Two sources:
//  • an external provider with a catalog → its models (catalog leads with
//    "Account default" (value ""), a valid pick — no forced choice);
//  • the Augmentum local engine → the installed models, placeholder-led
//    (a model IS required there; never auto-selected).
async function _populateModelControl(agent) {
  const sel = document.getElementById('cca-model');
  if (!sel) return;
  const provider = _composerProviders[agent];
  if (provider && (provider.models || []).length) {
    sel.innerHTML = provider.models
      .map((m) => `<option value="${escapeHtml(m.value || '')}">${escapeHtml(m.label || m.value || '')}</option>`)
      .join('');
    return;
  }
  let names = [];
  try {
    const models = await getModels();
    names = (models || [])
      .map((m) => (typeof m === 'string' ? m : (m.id || m.name || '')))
      .filter(Boolean);
  } catch (err) {
    console.debug('[Coder] composer models load failed', err);
  }
  sel.innerHTML = '<option value="">Choose a model…</option>' +
    names.map((n) => `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`).join('');
}

async function _onComposerTargetChange() {
  const target = document.getElementById('cca-target')?.value || '';
  if (target === 'machine') await _loadComposerExternalAgents();
  await _refreshComposerAgents();
}

function _onComposerAgentChange() {
  const agent = document.getElementById('cca-agent')?.value || '';
  const target = document.getElementById('cca-target')?.value || '';
  const provider = _composerProviders[agent];
  // Model control shows for the Augmentum local engine (installed models) and
  // for any model-targetable external agent (its provider catalog). Permission
  // is Claude-only.
  const showModel = agent === 'augmentum'
    || !!(provider && provider.model_targetable && (provider.models || []).length);
  document.getElementById('cca-model-wrap')?.classList.toggle('hidden', !showModel);
  document.getElementById('cca-perm-wrap')?.classList.toggle('hidden', agent !== 'claude');
  if (showModel) _populateModelControl(agent);
  // Derived locus label — informational, never a control.
  const locusHint = document.getElementById('cca-locus-hint');
  if (locusHint) {
    locusHint.textContent = target === 'machine' ? 'Runs on your own machine'
      : target.startsWith('ws:') ? 'Runs in a sandboxed workspace' : '';
  }
  _updateComposerHint();
  // The resume banner is Claude-only — drop it if we've moved off Claude.
  if (agent !== 'claude' && currentCoder().claudeResumeRunId) _setClaudeResume('', '');
}

// The offline-external stopgap: if "my machine" has no connected agent, show
// the paste-command that starts one (honest zero-infra path to real spawn).
function _updateComposerHint() {
  const hintEl = document.getElementById('cca-hint');
  if (!hintEl) return;
  const target = document.getElementById('cca-target')?.value || '';
  if (target === 'machine' && !_composerExternalAgents.length) {
    const task = (document.getElementById('cca-task')?.value || '').trim();
    const arg = task ? ` "${task.replace(/"/g, '\\"').slice(0, 80)}"` : '';
    hintEl.innerHTML = 'No agent is connected on your machine. Start one and it '
      + 'appears here — e.g. run <code>claude-aug' + escapeHtml(arg) + '</code> in your terminal.';
  } else {
    hintEl.textContent = '';
  }
}

// Point the composer at a specific agent/workspace (used by "Continue this
// run", which is Claude-only and must resume on the run's own workspace).
async function _selectComposerFor({ agent, wsId }) {
  const targetSel = document.getElementById('cca-target');
  if (targetSel && wsId) {
    const val = `ws:${wsId}`;
    if ([...targetSel.options].some((o) => o.value === val)) {
      targetSel.value = val;
      await _refreshComposerAgents();
    }
  }
  const agentSel = document.getElementById('cca-agent');
  if (agentSel && agent && [...agentSel.options].some((o) => o.value === agent && !o.disabled)) {
    agentSel.value = agent;
    _onComposerAgentChange();
  }
}

async function _initComposer() {
  await _loadComposerProviders();       // agent catalog + model capabilities
  await _loadComposerExternalAgents();
  await _loadComposerTargets();  // builds target + agent dropdowns; agent change
                                 // populates the Model control on demand
}

// Fire-and-list dispatch for the async engines (internal mission / harness
// assignment). The live run then appears in the runs list below with a pulse.
async function _dispatchAsync(statusEl, body, okMsg) {
  const btn = document.getElementById('cca-dispatch-btn');
  const taskEl = document.getElementById('cca-task');
  if (btn) btn.disabled = true;
  _claudeSetStatus(statusEl, 'Dispatching…', true);
  try {
    const r = await fetch('/api/coding/runs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const d = await r.json().catch(() => ({}));
    if (r.ok && d.ok !== false) {
      _claudeSetStatus(statusEl, okMsg, false);
      if (taskEl) taskEl.value = '';
      _refreshUnifiedRuns();
    } else {
      _claudeSetStatus(statusEl, d.error || 'Dispatch failed.', false);
    }
  } catch (err) {
    _claudeSetStatus(statusEl, 'Request failed.', false);
    console.debug('[Coder] dispatch failed', err);
  } finally {
    if (btn) btn.disabled = false;
  }
}

// The dispatch router — reads the composer and sends to the right engine.
async function _dispatchRun() {
  const statusEl = document.getElementById('cca-status');
  const task = (document.getElementById('cca-task')?.value || '').trim();
  if (!task) { _claudeSetStatus(statusEl, 'Type a task first.', false); return; }
  const target = document.getElementById('cca-target')?.value || '';
  const agent = document.getElementById('cca-agent')?.value || '';
  if (!target) { _claudeSetStatus(statusEl, 'Pick a target.', false); return; }
  if (!agent) { _claudeSetStatus(statusEl, 'Pick an agent.', false); return; }

  // Workspace target → internal locus.
  if (target.startsWith('ws:')) {
    const wsId = target.slice(3);
    if (agent === 'claude') {
      if (!currentCoder().claudeConnected) { _claudeSetStatus(statusEl, 'Connect Claude above first.', false); return; }
      const permission = document.getElementById('cca-permission')?.value || 'auto';
      const model = document.getElementById('cca-model')?.value || '';  // "" = account default
      await _streamClaudeRun({ task, wsId, permission, resumeRunId: currentCoder().claudeResumeRunId, model });
      return;
    }
    if (agent === 'augmentum') {
      const model = document.getElementById('cca-model')?.value || '';
      if (!model) { _claudeSetStatus(statusEl, 'Choose a model.', false); return; }
      await _dispatchAsync(statusEl,
        { driver: 'internal', workspace_id: wsId, task, model },
        'Dispatched — the mission is running. Watch it in the runs below.');
      return;
    }
    _claudeSetStatus(statusEl, 'Pick an agent.', false);
    return;
  }

  // My machine → external locus: assign to a live bridge agent session.
  if (target === 'machine') {
    if (!agent.startsWith('harness:')) {
      _claudeSetStatus(statusEl, 'No agent connected on your machine.', false);
      return;
    }
    const agentId = agent.slice('harness:'.length);
    await _dispatchAsync(statusEl,
      { driver: 'harness', agent_session_id: agentId, task },
      "Assigned — it'll pick this up at its next check-in.");
  }
}

function _wireAgentsModalOnce() {
  const modal = document.getElementById('coder-agents-modal');
  if (!modal || modal.dataset.wired === '1') return;
  modal.dataset.wired = '1';
  const close = document.getElementById('coder-agents-close');
  const connectBtn = document.getElementById('coder-claude-connect-btn');
  const discBtn = document.getElementById('coder-claude-disconnect-btn');
  const input = document.getElementById('coder-claude-token-input');
  const msg = document.getElementById('coder-claude-msg');
  const hide = () => modal.classList.add('hidden');
  close?.addEventListener('click', hide);
  modal.addEventListener('click', (e) => { if (e.target === modal) hide(); });
  connectBtn?.addEventListener('click', async () => {
    const token = (input?.value || '').trim();
    if (!token) { if (msg) msg.textContent = 'Paste your token first.'; return; }
    if (msg) msg.textContent = 'Connecting…';
    try {
      const r = await fetch('/api/coder/external/claude/token', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token }),
      });
      const d = await r.json();
      if (r.ok && d.connected) {
        if (input) input.value = '';
        if (msg) msg.textContent = d.warning || 'Connected.';
        _refreshClaudeAgentStatus();
      } else if (msg) {
        msg.textContent = d.error || 'Could not connect.';
      }
    } catch (err) {
      if (msg) msg.textContent = 'Request failed.';
      console.debug('[Coder] claude connect failed', err);
    }
  });
  discBtn?.addEventListener('click', async () => {
    try { await fetch('/api/coder/external/claude/token', { method: 'DELETE' }); }
    catch (err) { console.debug('[Coder] claude disconnect failed', err); }
    _refreshClaudeAgentStatus();
    if (msg) msg.textContent = 'Disconnected.';
  });
  // Composer: dispatch/stop + target/agent selectors.
  document.getElementById('cca-dispatch-btn')?.addEventListener('click', _dispatchRun);
  document.getElementById('cca-stop-btn')?.addEventListener('click', _stopClaudeRun);
  document.getElementById('cca-target')?.addEventListener('change', _onComposerTargetChange);
  document.getElementById('cca-agent')?.addEventListener('change', _onComposerAgentChange);
  document.getElementById('cca-task')?.addEventListener('input', _updateComposerHint);

  // Unified run-history controls.
  document.getElementById('coder-agents-runs-refresh')?.addEventListener('click', (e) => {
    e.preventDefault(); _refreshUnifiedRuns();
  });
  document.getElementById('coder-claude-resume-cancel')?.addEventListener('click', (e) => {
    e.preventDefault(); _setClaudeResume('', '');
  });
  // A whole row is the click target → open the windowed detail. Recover the
  // full row object (source/run_id/review_turn_id/session_id) from the cache.
  const onRowActivate = (e) => {
    const rowEl = e.target.closest('.cca-run-row--click');
    if (!rowEl) return;
    if (e.type === 'keydown' && e.key !== 'Enter' && e.key !== ' ') return;
    if (e.type === 'keydown') e.preventDefault();
    const row = currentCoder().unifiedRunsById.get(rowEl.dataset.runId);
    if (row) _openAgentDetail(row);
  };
  for (const id of ['cca-internal-list', 'cca-external-list']) {
    const el = document.getElementById(id);
    el?.addEventListener('click', onRowActivate);
    el?.addEventListener('keydown', onRowActivate);
  }
}

function _updateAgentsTile(ws) {
  const tile = document.getElementById('coder-agents-tile');
  if (!tile) return;
  tile.classList.toggle('hidden', !ws);
  _wireAgentsModalOnce();
  if (!ws || tile.dataset.wired === '1') return;
  tile.dataset.wired = '1';
  tile.addEventListener('click', () => {
    const modal = document.getElementById('coder-agents-modal');
    if (!modal) return;
    modal.classList.remove('hidden');
    _refreshClaudeAgentStatus();
    _initComposer();
    _refreshUnifiedRuns();
  });
}

function _updateBugFinderTile(ws) {
  const tile = document.getElementById('coder-bug-finder-tile');
  if (!tile) return;
  // Tile is visible on every workspace as a manual entry point. The
  // launcher only auto-opens on the very first entry into a workspace
  // tagged `kind=bug_finder` (see _maybeOpenBugFinderForWorkspace);
  // everywhere else this tile is the discoverable launcher.
  const visible = !!ws;
  tile.classList.toggle('hidden', !visible);
  if (!visible || tile.dataset.wired === '1') return;
  tile.dataset.wired = '1';
  tile.addEventListener('click', async () => {
    const wsId = currentCoder().workspaceId;
    if (!wsId) return;
    try {
      const mod = await import('./bug-finder.js');
      mod.launchForWorkspace({
        workspaceId: wsId,
        primaryModel: app?.state?.currentModel || '',
        verifierModel: _activeVerifierModel || '',
        openLauncher: false,
      });
    } catch (err) {
      console.debug('[Coder] bug-finder open failed', err);
    }
  });
}

// localStorage key for the last-active workspace id. Restored on init so a
// page refresh lands the user back on the workspace they were working in,
// not on whichever workspace happens to sort first in the API response.
// Without this, chat history + terminal + file tree would silently re-bind
// to a different workspace after refresh — the user perceives this as
// "chat/terminal got disconnected" because the active workspace doesn't
// match what they remember leaving open.
const _ACTIVE_WORKSPACE_STORAGE_KEY = 'augmentum.coder.activeWorkspaceId';

function _persistActiveWorkspaceId(id) {
  try {
    if (id) localStorage.setItem(_ACTIVE_WORKSPACE_STORAGE_KEY, id);
    else localStorage.removeItem(_ACTIVE_WORKSPACE_STORAGE_KEY);
  } catch { /* private mode / quota — silent fallback to in-memory only */ }
}

function _recallActiveWorkspaceId() {
  try { return localStorage.getItem(_ACTIVE_WORKSPACE_STORAGE_KEY) || ''; }
  catch { return ''; }
}
// Coder loop strategy is fixed to "native" in the UI — the lean
// Claude-Code/Qwen-Code parity loop and the single source of truth for
// loop guards (see augmentum/modes/coder/README.md). The hybrid/canonical/
// legacy loops are frozen comparison/rollback paths reachable only via the
// AUGMENTUM_CODER_STRATEGY backend env var; the UI no longer sends the
// X-Augmentum-Coder-Strategy header, so the backend default/env decides.
// Module-scoped flag — ensures the coder:turn-reviewed listener is
// registered exactly once per session even though onReviewPending
// may fire many times. Alternative (listener in init()) would be
// cleaner but init runs before _scheduleFileTreeRefresh is in scope
// with an active workspace. Lazy registration wins.

// Pending chat-composer attachments. Populated by the drop handler on
// #coder-conversation; drained on send. Each entry is the descriptor
// shape returned by coder-attachments.js#ingestFile.

// Editor file tracking (slide-in panel)

// Static file preview state. Distinct slot from currentCoder().previewInfo (dev-server
// proxy) so the 5s _refreshPorts poll can't overwrite it. When non-null,
// _renderPreviewPane shows this URL in the iframe and skips the
// dev-server "kick back to terminal if not_published" logic entirely.
// Cleared on workspace switch (via _resetPreviewState), on explicit
// close (× button on preview header), or when opening a different file.

// Conversation-first components

// Last user prompt the agent ran. Captured so the recoverable-error
// pill's "Try Again" button can re-fire the SAME turn after a transient
// backend failure (429 / 5xx / network blip exhausted the retry
// budget) without forcing the user to retype.

// Tooling profile catalog is fetched from the server on demand and cached
// for the page lifetime. Single source of truth lives in
// augmentum/coder/profiles.py — adding a profile there propagates here
// automatically. The hardcoded fallback exists ONLY for the brief window
// between page load and the first fetch returning (or in the offline /
// API-down case) so the dropdown is never empty.
const _TOOLING_PROFILE_FALLBACK = [
  { id: 'standard', label: 'Standard', description: 'Fast baseline for most Python, JS, Go, and Rust work.' },
  { id: 'power',    label: 'Power',    description: 'Adds process/network inspection, uv/pipx, pnpm/yarn, and build/debug tools.' },
  { id: 'browser',  label: 'Browser/Test', description: 'Power profile; browser automation via the shared browser sidecar service.' },
];

let _toolingProfileCache = null;          // Promise<Profile[]> | null
let _toolingProfileCacheValue = null;     // Profile[] | null — sync access for option render

async function _loadToolingProfiles() {
  if (_toolingProfileCache) return _toolingProfileCache;
  _toolingProfileCache = (async () => {
    try {
      const resp = await fetch('/api/coder/tooling-profiles');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const profiles = Array.isArray(data?.profiles) && data.profiles.length
        ? data.profiles
        : _TOOLING_PROFILE_FALLBACK;
      _toolingProfileCacheValue = profiles;
      return profiles;
    } catch (err) {
      console.warn('Tooling profile fetch failed, using fallback', err);
      _toolingProfileCacheValue = _TOOLING_PROFILE_FALLBACK;
      return _TOOLING_PROFILE_FALLBACK;
    }
  })();
  return _toolingProfileCache;
}

function _toolingProfileOptions(selected = 'browser') {
  // Synchronous render — uses whatever's in cache. Callers that need
  // the freshest list await ``_loadToolingProfiles()`` before rendering
  // the modal that owns the dropdown.
  const profiles = _toolingProfileCacheValue || _TOOLING_PROFILE_FALLBACK;
  return profiles.map((profile) => `
    <option value="${escapeHtml(profile.id)}" ${profile.id === selected ? 'selected' : ''}>
      ${escapeHtml(profile.label)} - ${escapeHtml(profile.description || '')}
    </option>
  `).join('');
}

export function getActiveWorkspaceId() { return currentCoder().workspaceId || ''; }
let _initialized = false;
let _panelAutoClosed = false;  // one-shot: collapse left panel on first coder-mode entry

// ---------------------------------------------------------------------------
// Tool result preview rendering (2026-04-20)
// ---------------------------------------------------------------------------
//
// Pre-2026-04-20 the terminal only echoed "✖ failed" on failures and
// showed nothing on success. That left users blind to what the agent
// actually did — Qwen 3.6 traces showed 15 shell_exec calls, silent
// stop, empty "Done" banner. Claude Code streams tool results inline
// for all tools; this is our version of that.
//
// Two call sites render tool results into the terminal:
//   * CoderStream's onToolResult (conversation-driven turns)
//   * _runAgentInTerminal's raw SSE loop (// prefix in terminal)
// Both route through _renderToolResultTo(echo, result).

const _READ_TOOLS_FOR_PREVIEW = new Set([
  'file_read', 'dir_tree', 'file_list',
  'code_grep', 'find_files', 'code_search',
  'env_info', 'doc_search', 'doc_fetch',
]);
const _WRITE_TOOLS_FOR_PREVIEW = new Set([
  'file_write', 'code_edit', 'code_edit_batch',
]);

const _ANSI_DIM = '\x1b[90m';
const _ANSI_RED = '\x1b[31m';
const _ANSI_GRN = '\x1b[32m';
const _ANSI_YLW = '\x1b[33m';
const _ANSI_RESET = '\x1b[0m';

function _renderToolResultTo(echo, result) {
  const preview = (result.output_preview || '').replace(/\r?\n/g, '\n');
  const tool = result.tool || '';
  const success = result.success !== false;

  // Preemptive refusal — distinct signal from a real failure.
  if (result.preemptive_refusal) {
    const n = result.prior_count || 0;
    echo(
      `   ${_ANSI_YLW}⊘ refused (already called ×${n}; see earlier results)` +
      `${_ANSI_RESET}\r\n`,
    );
    return;
  }

  // Batch duplicate — compact hint (result content lives in the
  // canonical tool_result earlier in the same batch).
  if (result.batch_duplicate) {
    echo(
      `   ${_ANSI_DIM}↺ duplicate of earlier call (dedup'd)${_ANSI_RESET}\r\n`,
    );
    return;
  }

  // Fanout-dropped — compact hint.
  if (result.fanout_dropped) {
    echo(
      `   ${_ANSI_YLW}⊖ skipped (read fanout exceeded)${_ANSI_RESET}\r\n`,
    );
    return;
  }

  // Scratch-externalised — annotate the scratch path so users know
  // where to find the full output if they want to read it.
  let scratchNote = '';
  if (result.scratch_path) {
    const shortPath = result.scratch_path.replace(
      '/workspace/.augmentum/scratch/', '',
    );
    const kb = result.scratch_size
      ? ` (${Math.round(result.scratch_size / 1024)}KB)` : '';
    scratchNote = (
      `   ${_ANSI_DIM}↳ full output${kb} in scratch/${shortPath}${_ANSI_RESET}\r\n`
    );
  }

  // Failure path — always show the error preview in red, regardless
  // of tool type. Compact to 8 lines so long tracebacks don't dominate.
  if (!success) {
    if (!preview) {
      echo(`${_ANSI_RED}   ✖ failed${_ANSI_RESET}\r\n`);
      return;
    }
    const lines = preview.split('\n');
    const MAX = 8;
    const shown = lines.slice(0, MAX).join('\r\n   ');
    const more = lines.length > MAX
      ? `\r\n   ${_ANSI_DIM}... ${lines.length - MAX} more lines${_ANSI_RESET}`
      : '';
    echo(`${_ANSI_RED}   ${shown}${more}${_ANSI_RESET}\r\n`);
    return;
  }

  // Success path — tool-specific rendering.
  if (!preview) {
    // Some tools report success with no output (e.g. task_list mark
    // complete). Give a minimal check — silence was the bug we're
    // fixing. Intentionally terse.
    echo(`${_ANSI_GRN}   ✓ ok${_ANSI_RESET}\r\n`);
    if (scratchNote) echo(scratchNote);
    return;
  }

  if (tool === 'shell_exec' || tool === 'shell_read') {
    // Shell output is usually the deliverable — show up to 1200 chars
    // raw (dim gray). Multi-line preserved.
    const clipped = preview.slice(0, 1200);
    const more = preview.length > 1200
      ? `\r\n   ${_ANSI_DIM}... (truncated; full in history)${_ANSI_RESET}`
      : '';
    echo(
      `${_ANSI_DIM}${clipped.replace(/\n/g, '\r\n')}${more}${_ANSI_RESET}\r\n`,
    );
    if (scratchNote) echo(scratchNote);
    return;
  }

  if (_READ_TOOLS_FOR_PREVIEW.has(tool)) {
    // Structured read — first 10 lines with trailing-count hint.
    const lines = preview.split('\n');
    const MAX = 10;
    const shown = lines.slice(0, MAX).join('\r\n   ');
    const more = lines.length > MAX
      ? `\r\n   ${_ANSI_DIM}... ${lines.length - MAX} more lines${_ANSI_RESET}`
      : '';
    echo(`${_ANSI_DIM}   ${shown}${more}${_ANSI_RESET}\r\n`);
    if (scratchNote) echo(scratchNote);
    return;
  }

  if (_WRITE_TOOLS_FOR_PREVIEW.has(tool)) {
    // Write confirmation — backend's first line is usually "Wrote X
    // bytes to /path" or "Applied N changes". Checkpoint (if any)
    // renders on its own line.
    const firstLine = (preview.split('\n')[0] || '').slice(0, 140);
    echo(`${_ANSI_GRN}   ✓ ${firstLine}${_ANSI_RESET}\r\n`);
    if (result.checkpoint) {
      echo(
        `   ${_ANSI_DIM}↳ checkpoint ${String(result.checkpoint).slice(0, 7)}${_ANSI_RESET}\r\n`,
      );
    }
    if (scratchNote) echo(scratchNote);
    return;
  }

  if (tool === 'test_run') {
    // First line is the structured "Passed: X Failed: Y Errors: Z".
    const firstLine = (preview.split('\n')[0] || '').slice(0, 140);
    const icon = success ? '✓' : '✖';
    const color = success ? _ANSI_GRN : _ANSI_RED;
    echo(`${color}   ${icon} ${firstLine}${_ANSI_RESET}\r\n`);
    if (scratchNote) echo(scratchNote);
    return;
  }

  if (tool === 'task_list') {
    // Plan transparency: render the whole list, not just the first
    // line. Backend emits "Task list updated — N item(s):" followed by
    // one "  [x|~| ] content" line per task — clipping at split('\n')[0]
    // hid every task and showed only the count. Each status marker
    // gets a color so completed/in-progress/pending read at a glance.
    // Cap is generous (40 lines) so multi-phase plans stay visible
    // without flooding the terminal on pathological cases.
    const lines = preview.split('\n');
    const MAX = 40;
    const head = lines[0] || '';
    echo(`${_ANSI_GRN}   ✓ ${head}${_ANSI_RESET}\r\n`);
    const rest = lines.slice(1, MAX + 1);
    for (const raw of rest) {
      // Match "  [x] foo", "  [~] foo", "  [ ] foo" with any leading whitespace.
      const m = raw.match(/^\s*\[([x~ ])\]\s*(.*)$/);
      if (m) {
        const status = m[1];
        const body = m[2];
        const color = status === 'x' ? _ANSI_GRN
          : status === '~' ? _ANSI_YLW
          : _ANSI_DIM;
        const marker = status === 'x' ? '[x]'
          : status === '~' ? '[~]'
          : '[ ]';
        echo(`${color}     ${marker} ${body}${_ANSI_RESET}\r\n`);
      } else if (raw.trim()) {
        // Free-form lines (rare) — dim them so they don't compete.
        echo(`${_ANSI_DIM}     ${raw.trim()}${_ANSI_RESET}\r\n`);
      }
    }
    if (lines.length > MAX + 1) {
      const remaining = lines.length - (MAX + 1);
      echo(`   ${_ANSI_DIM}... ${remaining} more line(s)${_ANSI_RESET}\r\n`);
    }
    if (scratchNote) echo(scratchNote);
    return;
  }

  // Default: first line of preview. Covers ask_user, git, and any
  // future tool that doesn't match the groups above.
  const firstLine = (preview.split('\n')[0] || '').slice(0, 140);
  echo(`${_ANSI_DIM}   ${firstLine}${_ANSI_RESET}\r\n`);
  if (scratchNote) echo(scratchNote);
}

// ---------------------------------------------------------------------------
// Coder-Scoped Theme System
// Isolated from the app-wide theme. Persisted separately in localStorage.
// ---------------------------------------------------------------------------
// 'app' inherits the app-wide theme (light/sepia/midnight/dark) instead of
// forcing a coder-only dark palette — without it, entering coder mode in
// light mode slammed the whole surface dark and looked broken. Default for
// users who never explicitly picked a coder theme; an explicit pick sticks.
const CODER_THEMES = ['app', 'satin', 'cyber', 'aurora', 'matrix'];
const CODER_THEME_LABELS = { app: 'App', satin: 'Satin', cyber: 'Cyber', aurora: 'Aurora', matrix: 'Matrix' };
const CODER_THEME_KEY = 'augmentum-coder-theme';
let _coderTheme = localStorage.getItem(CODER_THEME_KEY) || 'app';
if (!CODER_THEMES.includes(_coderTheme)) _coderTheme = 'app';

function _applyCoderTheme() {
  const appEl = document.getElementById('app');
  if (appEl) {
    // 'app' = no override attribute → the app theme's variables flow through.
    if (_coderTheme === 'app') appEl.removeAttribute('data-coder-theme');
    else appEl.setAttribute('data-coder-theme', _coderTheme);
  }
  const label = document.getElementById('coder-theme-label');
  if (label) label.textContent = CODER_THEME_LABELS[_coderTheme] || _coderTheme;
  // Refresh terminal colors if available
  try { Terminal.updateTheme(); } catch {}
}

function _removeCoderTheme() {
  const appEl = document.getElementById('app');
  if (appEl) appEl.removeAttribute('data-coder-theme');
}

function _cycleCoderTheme() {
  const idx = CODER_THEMES.indexOf(_coderTheme);
  _coderTheme = CODER_THEMES[(idx + 1) % CODER_THEMES.length];
  localStorage.setItem(CODER_THEME_KEY, _coderTheme);
  _applyCoderTheme();
  showToast(`Coder Theme: ${CODER_THEME_LABELS[_coderTheme]}`, 'success');
}


function _showMobileWorkbenchSurface() {
  // Below the 1024 side-by-side band the terminal/preview pane is only shown
  // when it's the active pane — flip to it. (No-op above 1024 where panes
  // already coexist.)
  if (window.innerWidth >= 1024) return;
  _setCoderPane(currentCoder().previewInfo.state !== 'not_published' && currentCoder().activeWorkbenchTab === 'preview' ? 'preview' : 'terminal');
}

function _setWorkbenchTab(tab) {
  const wantsPreview = tab === 'preview' &&
    (currentCoder().previewInfo.state !== 'not_published' || !!currentCoder().filePreview);
  currentCoder().activeWorkbenchTab = wantsPreview ? 'preview' : 'terminal';
  currentCoder().dom.workbenchTerminalTab?.classList.toggle('active', currentCoder().activeWorkbenchTab === 'terminal');
  currentCoder().dom.workbenchPreviewTab?.classList.toggle('active', currentCoder().activeWorkbenchTab === 'preview');
  currentCoder().dom.terminalPane?.classList.toggle('hidden', currentCoder().activeWorkbenchTab !== 'terminal');
  currentCoder().dom.previewPane?.classList.toggle('hidden', currentCoder().activeWorkbenchTab !== 'preview');
  currentCoder().dom.previewReloadBtn?.classList.toggle('hidden', currentCoder().activeWorkbenchTab !== 'preview' || !currentCoder().previewInfo.ready);
  currentCoder().dom.previewOpenExternalBtn?.classList.toggle('hidden', currentCoder().activeWorkbenchTab !== 'preview' || !currentCoder().previewInfo.ready);
  currentCoder().dom.previewSaveBtn?.classList.toggle('hidden', currentCoder().activeWorkbenchTab !== 'preview' || !currentCoder().previewInfo.ready);
  if (currentCoder().activeWorkbenchTab === 'terminal' && currentCoder().terminalId) {
    requestAnimationFrame(() => Terminal.fit(currentCoder().terminalId));
  }
  // Keep the top pane-switch's terminal/preview active state honest when the
  // workbench tab changes outside _setCoderPane (e.g. preview-ready auto-flip,
  // desktop header tab clicks). Only when the terminal side is the active pane
  // so we don't steal "active" away from chat.
  const layoutPane = (currentCoder().dom.coderLayout || document.getElementById('coder-layout'))?.getAttribute('data-coder-pane');
  if (layoutPane && layoutPane !== 'chat') _syncPaneControls(currentCoder().activeWorkbenchTab);
  _updatePreviewFrameActivity();
}

function _resetWorkbenchTabForHiddenCoder() {
  currentCoder().activeWorkbenchTab = 'terminal';
  currentCoder().dom.workbenchTerminalTab?.classList.add('active');
  currentCoder().dom.workbenchPreviewTab?.classList.remove('active');
  currentCoder().dom.terminalPane?.classList.add('hidden');
  currentCoder().dom.previewPane?.classList.add('hidden');
  currentCoder().dom.previewReloadBtn?.classList.add('hidden');
  currentCoder().dom.previewOpenExternalBtn?.classList.add('hidden');
  currentCoder().dom.previewSaveBtn?.classList.add('hidden');
  currentCoder().dom.previewSavePrompt?.classList.add('hidden');
  _updatePreviewFrameActivity();
}

// --- Preview iframe lifecycle (visibility-gated) ----------------------------
// A CSS-hidden iframe KEEPS EXECUTING its document's JS — and the preview
// runs on the same renderer main thread as the whole coder UI (same-site in
// every mode: the proxy path is same-origin, and the "isolated origin" is
// just another port, which browsers treat as the same site for process
// allocation). A dev app with a busy loop (agent-mid-edit code, a game's
// rAF loop) therefore froze the page even with the Terminal tab active and
// no agent request running — in EVERY connected browser at once. Sandbox
// flags are a security boundary, not a scheduling one.
//
// The fix: only let the preview document exist while it's actually on
// screen. Anything else parks the frame at about:blank and remembers how to
// come back (dataset.previewSameOriginUrl already survives — the expiry
// re-mint path uses the same marker). Resuming goes through
// _renderPreviewPane → _setPreviewSrc, which re-mints the isolation token —
// so a park/resume cycle also transparently handles token expiry.
//
// TRADE-OFF (deliberate): the previewed app's in-page state resets when the
// user tabs away and back. That's inherent to unloading; the dev server
// itself keeps running in the workspace, untouched.

/** True when the preview iframe is actually visible to the user right now:
 *  the workbench tab is "preview", the mobile pane layout isn't covering it
 *  with chat, and the browser tab itself is foregrounded. */
function _previewFrameVisible() {
  if (currentCoder().activeWorkbenchTab !== 'preview') return false;
  if (document.hidden) return false;
  // Below the side-by-side band, data-coder-pane="chat" covers the
  // workbench entirely even though currentCoder().activeWorkbenchTab stays "preview".
  if (window.innerWidth < 1024) {
    const layoutPane = (currentCoder().dom.coderLayout || document.getElementById('coder-layout'))
      ?.getAttribute('data-coder-pane');
    if (layoutPane === 'chat') return false;
  }
  return true;
}

/** Unload the frame and mark that there's content to restore. The
 *  same-origin URL marker (dataset.previewSameOriginUrl) is left in place —
 *  it's how the resume path (and the expiry re-mint path) find the way back. */
function _parkPreviewFrame(frame) {
  frame.dataset.previewParked = '1';
  const src = frame.getAttribute('src') || '';
  if (src && src !== 'about:blank') frame.src = 'about:blank';
}

/** Park (unload) or resume the preview iframe to match actual visibility.
 *  Called from every place visibility can change: workbench tab switch,
 *  mobile pane switch, mode exit, and document visibilitychange. */
function _updatePreviewFrameActivity() {
  const frame = currentCoder().dom.previewFrame;
  if (!frame) return;
  const visible = _previewFrameVisible();
  const src = frame.getAttribute('src') || '';
  const loaded = !!src && src !== 'about:blank';
  if (!visible && loaded) {
    _parkPreviewFrame(frame);
  } else if (visible && frame.dataset.previewParked === '1') {
    delete frame.dataset.previewParked;
    // Re-derive the right URL (file preview vs dev server) through the
    // normal renderer — it re-mints the isolation token as a side effect.
    _renderPreviewPane();
  }
}

// --- Unified pane navigation (Chat / Terminal / Preview) -------------------
// The single entry point for switching which pane is active below the 1024px
// side-by-side band. Records the choice as data-coder-pane on #coder-layout
// (the CSS authority for the single-active-pane layout) and keeps BOTH the
// top pane-switch and the legacy bottom emoji bar in sync, so neither being
// clipped or missing can strand the user. Above 1024px the panes coexist, so
// this only flips terminal⇄preview and leaves chat visible.
function _setCoderPane(pane) {
  const layout = currentCoder().dom.coderLayout || document.getElementById('coder-layout');
  if (!layout) return;
  const wantsPreview = pane === 'preview' &&
    (currentCoder().previewInfo.state !== 'not_published' || !!currentCoder().filePreview);
  const onTerminalSide = pane === 'terminal' || wantsPreview;
  const active = onTerminalSide ? (wantsPreview ? 'preview' : 'terminal') : 'chat';

  // CSS authority for the ≤1023 single-active layout.
  layout.setAttribute('data-coder-pane', active);
  // Keep the legacy ≤767 mob-* mechanism in agreement (belt-and-suspenders).
  const conv = document.getElementById('coder-conversation');
  const termWrap = document.getElementById('coder-terminal-wrapper');
  conv?.classList.toggle('mob-hidden', onTerminalSide);
  termWrap?.classList.toggle('mob-visible', onTerminalSide);

  if (onTerminalSide) {
    _setWorkbenchTab(wantsPreview ? 'preview' : 'terminal');
    _hideEditorSplit();
  }

  // Extra-keys bar (Ctrl/Esc/arrows) is terminal-only — not chat or preview.
  const extraKeys = document.getElementById('coder-extra-keys');
  const mainArea = document.querySelector('.main-area');
  const showExtraKeys = pane === 'terminal';
  if (extraKeys) extraKeys.style.display = showExtraKeys ? '' : 'none';
  if (mainArea) mainArea.classList.toggle('coder-has-extra-keys', showExtraKeys);

  _syncPaneControls(active);

  if (onTerminalSide && !wantsPreview && currentCoder().terminalId) {
    requestAnimationFrame(() => { Terminal.fit(currentCoder().terminalId); Terminal.focus(currentCoder().terminalId); });
  } else if (pane === 'chat') {
    const scroll = document.getElementById('coder-conv-scroll');
    if (scroll) requestAnimationFrame(() => { scroll.scrollTop = scroll.scrollHeight; });
    document.getElementById('coder-input')?.focus();
  }

  // The chat pane covers the workbench below 1024px without touching
  // currentCoder().activeWorkbenchTab — re-evaluate whether the preview iframe should
  // keep running (onTerminalSide paths already did this via _setWorkbenchTab).
  _updatePreviewFrameActivity();
}

// Reflect the active pane on every control that can switch panes: the top
// pane-switch (chat/terminal/preview) and the bottom emoji bar (chat vs
// terminal — preview maps to terminal there since it has no preview button).
function _syncPaneControls(active) {
  currentCoder().dom.paneSwitch?.querySelectorAll('.coder-pane-btn').forEach((btn) => {
    const on = btn.dataset.pane === active;
    btn.classList.toggle('active', on);
    btn.setAttribute('aria-selected', on ? 'true' : 'false');
  });
  const emojiActive = active === 'chat' ? 'chat' : 'terminal';
  currentCoder().dom.mobileTabs?.querySelectorAll('.coder-mob-tab').forEach((btn) => {
    const tab = btn.dataset.coderTab;
    if (tab === 'chat' || tab === 'terminal') {
      btn.classList.toggle('active', tab === emojiActive);
    }
  });
}

// Sandbox flags the iframe carries when the preview is confirmed to be
// loading from a different origin (isolated mode). "allow-same-origin"
// is safe in that case because "same-origin" inside the iframe refers to
// the isolated port, not Augmentum's main UI. In same-origin fallback
// mode we strip it so a malicious npm dep can't reach the parent's
// session cookies or /api/* surface.
const _SANDBOX_BASE = 'allow-scripts allow-forms allow-modals allow-popups';
const _SANDBOX_ISOLATED = _SANDBOX_BASE + ' allow-same-origin';

// Per-workspace localStorage key — when set to "1", the same-origin
// fallback path keeps allow-same-origin on the iframe, restoring the
// pre-hardening behaviour. Provided so users who can't deploy the
// isolated-origin Caddy/compose setup can still opt a specific
// workspace they trust back into the full dev-server experience
// (HMR cookies, in-iframe fetch credentials). Granular per workspace
// so trusting one project doesn't loosen the others. No server
// component: this is a browser-local affordance.
function _previewSameOriginTrustKey(workspaceId) {
  return workspaceId ? `coder.preview.trustSameOrigin.${workspaceId}` : '';
}

// Runtime source of truth for per-workspace same-origin trust. localStorage
// is the persistence layer, but it can silently fail — private mode, or a
// self-signed-cert LAN origin where the browser partitions/blocks site
// storage. When the write was swallowed, the flag was unreadable, so the
// sandbox stayed tight and the trust banner never disappeared after the user
// clicked "Trust this workspace". This Set guarantees trust takes effect for
// the session regardless; localStorage just makes it survive a reload.
const _previewTrustedWorkspaces = new Set();

function _isPreviewSameOriginTrusted(workspaceId) {
  if (!workspaceId) return false;
  if (_previewTrustedWorkspaces.has(workspaceId)) return true;
  const key = _previewSameOriginTrustKey(workspaceId);
  try {
    if (localStorage.getItem(key) === '1') {
      _previewTrustedWorkspaces.add(workspaceId);  // hydrate runtime cache
      return true;
    }
  } catch { /* storage blocked — the runtime Set is authoritative */ }
  return false;
}

function _applySandbox(frame, mode) {
  if (!frame) return;
  // mode 'isolated' → isolated origin (already safe). mode 'same-origin'
  // → loose only when the user has explicitly trusted THIS workspace via
  // the per-workspace toggle; otherwise tighten.
  let useLoose = mode === 'isolated';
  if (!useLoose && mode === 'same-origin' && _isPreviewSameOriginTrusted(currentCoder().workspaceId)) {
    useLoose = true;
  }
  const next = useLoose ? _SANDBOX_ISOLATED : _SANDBOX_BASE;
  if (frame.getAttribute('sandbox') !== next) {
    frame.setAttribute('sandbox', next);
  }
  frame.dataset.previewMode = mode;
  frame.dataset.previewSandbox = useLoose ? 'loose' : 'tight';
  // Reflect the sandbox state in the surrounding UI so the user knows
  // why their dev server is failing (tight) or can revoke trust (loose).
  _refreshPreviewTrustUi();
}

function _refreshPreviewTrustUi() {
  const banner = document.getElementById('coder-preview-trust-banner');
  const status = document.getElementById('coder-preview-trust-status');
  if (!banner || !status) return;
  const frame = currentCoder().dom.previewFrame;
  const mode = frame?.dataset?.previewMode || '';
  const sandbox = frame?.dataset?.previewSandbox || '';
  // The banner is the "you're broken, click to fix" call-to-action —
  // only when we're in tight same-origin mode (the cohort that hits
  // the CORS / 401 cascade the user reported).
  const showBanner = mode === 'same-origin' && sandbox === 'tight' && !!currentCoder().workspaceId;
  banner.classList.toggle('hidden', !showBanner);
  // The status row is the "you're trusted, here's the revoke" reminder
  // — only when we're loose AND the trust flag is what made it loose
  // (isolated-mode iframes don't need the trust flag).
  const isolated = mode === 'isolated';
  const showStatus = !isolated && sandbox === 'loose' && !!currentCoder().workspaceId
    && _isPreviewSameOriginTrusted(currentCoder().workspaceId);
  status.classList.toggle('hidden', !showStatus);
}

function _setPreviewSameOriginTrust(workspaceId, trusted) {
  if (!workspaceId) return;
  // Runtime Set first so trust applies immediately even when storage is
  // unavailable; localStorage is best-effort persistence across reloads.
  if (trusted) _previewTrustedWorkspaces.add(workspaceId);
  else _previewTrustedWorkspaces.delete(workspaceId);
  const key = _previewSameOriginTrustKey(workspaceId);
  try {
    if (trusted) localStorage.setItem(key, '1');
    else localStorage.removeItem(key);
  } catch { /* best-effort — the runtime Set already holds the truth */ }
}

// Apply a sandbox change RELIABLY by replacing the iframe element.
//
// Mutating the `sandbox` attribute on a live iframe and reloading does
// not consistently re-apply the new policy across browser engines — the
// already-loaded document keeps the sandbox it was born with, so the old
// about:blank→src dance left "Trust this workspace" looking like it did
// nothing (the dev-server assets that need allow-same-origin still
// failed). A freshly-created iframe is born with the new sandbox, so the
// policy is guaranteed to take effect on the next navigation. Returns the
// new frame element (or null if there was nothing to recreate).
function _recreatePreviewFrame(mode) {
  const old = currentCoder().dom.previewFrame;
  if (!old || !old.parentNode) return null;
  const curSrc = old.getAttribute('src') || '';
  const src = (curSrc && curSrc !== 'about:blank')
    ? curSrc
    : (currentCoder().activePreviewUrl || currentCoder().previewInfo.primary_url || '');
  const next = document.createElement('iframe');
  next.id = 'coder-preview-frame';
  next.className = old.className;        // preserve hidden/visible state
  next.title = old.title || 'Coder preview';
  // Carry the same-origin URL marker the expiry-remint path reads.
  if (old.dataset.previewSameOriginUrl) {
    next.dataset.previewSameOriginUrl = old.dataset.previewSameOriginUrl;
  }
  next.dataset.previewMode = mode;
  old.parentNode.replaceChild(next, old);
  currentCoder().dom.previewFrame = next;
  // Re-install the (one-time-guarded) debug load listeners on the new node.
  _previewFrameListenersInstalled = false;
  _installPreviewFrameListeners();
  // Born with the correct sandbox for the (possibly just-trusted) mode.
  _applySandbox(next, mode);
  if (src) next.src = src;
  return next;
}

function _initPreviewTrustControls() {
  const trustBtn = document.getElementById('coder-preview-trust-btn');
  const revokeBtn = document.getElementById('coder-preview-trust-revoke');
  // Idempotent re-entry guard since _onEnterCoderMode fires repeatedly.
  if (trustBtn && trustBtn.dataset.wired !== '1') {
    trustBtn.dataset.wired = '1';
    trustBtn.addEventListener('click', () => {
      if (!currentCoder().workspaceId) return;
      _setPreviewSameOriginTrust(currentCoder().workspaceId, true);
      // Recreate the iframe so it is born with allow-same-origin — the
      // only reliable way to make the looser sandbox take effect — then
      // it re-navigates and the dev-server fetches that failed under the
      // null origin get a second chance under the real origin.
      _recreatePreviewFrame('same-origin');
      showToast('Workspace trusted — preview can now load dev-server assets', 'success');
    });
  }
  if (revokeBtn && revokeBtn.dataset.wired !== '1') {
    revokeBtn.dataset.wired = '1';
    revokeBtn.addEventListener('click', () => {
      if (!currentCoder().workspaceId) return;
      _setPreviewSameOriginTrust(currentCoder().workspaceId, false);
      const mode = currentCoder().dom.previewFrame?.dataset?.previewMode || 'same-origin';
      // Recreate so the tighter sandbox is genuinely re-applied (same
      // reason as trusting — a live attribute swap is unreliable).
      _recreatePreviewFrame(mode);
      showToast('Trust revoked — preview is now tightly sandboxed', 'info');
    });
  }
}

// ── "Chime when done" — opt-in audio cue on turn completion ──────────────
// When enabled, a short chime plays at the end of a run IF this tab is in
// the background, so a user who stepped away to another tab knows the run
// is ready without watching it. Device-local chrome preference
// (localStorage), like the other coder behavior toggles. Reuses the shared
// notification-sound module so there's a single audio path to maintain.
const _CHIME_WHEN_DONE_KEY = 'coder.chimeWhenDone';

function _isChimeWhenDoneEnabled() {
  try { return localStorage.getItem(_CHIME_WHEN_DONE_KEY) === '1'; }
  catch { return false; }
}

function _setChimeWhenDone(on) {
  try {
    if (on) localStorage.setItem(_CHIME_WHEN_DONE_KEY, '1');
    else localStorage.removeItem(_CHIME_WHEN_DONE_KEY);
  } catch { /* storage blocked — toggle is best-effort for the session */ }
}

function _updateChimeButton() {
  const btn = document.getElementById('coder-chime-btn');
  if (!btn) return;
  const on = _isChimeWhenDoneEnabled();
  btn.style.color = on ? 'var(--accent, #6cf)' : 'var(--text-muted)';
  btn.style.opacity = on ? '' : '0.55';
  btn.title = on
    ? 'Chime when done: ON — plays a sound when a run finishes while this tab is in the background. Click to turn off.'
    : "Chime when done: OFF — click to play a sound when a run finishes while you're on another tab.";
}

function _initChimeControl() {
  const btn = document.getElementById('coder-chime-btn');
  if (!btn) return;
  if (btn.dataset.wired === '1') { _updateChimeButton(); return; }
  btn.dataset.wired = '1';
  btn.addEventListener('click', () => {
    _setChimeWhenDone(!_isChimeWhenDoneEnabled());
    _updateChimeButton();
  });
  _updateChimeButton();
}

// Play the completion chime when enabled AND the tab is backgrounded — no
// point chiming if the user is already watching the run finish.
function _maybeChimeOnDone() {
  if (!_isChimeWhenDoneEnabled() || !document.hidden) return;
  import('./notification-sound.js')
    .then((m) => m.playNotificationSound('chime'))
    .catch(() => { /* audio unavailable / blocked — stay silent */ });
}

// Set the preview iframe src, transparently handling origin isolation.
//
// When coder_preview_isolation_enabled is True server-side, this mints a
// one-time token and routes the iframe to a different origin (different
// port on the same host) so the preview content cannot reach Augmentum's
// /api/* with the user's session cookies. When isolation is disabled, the
// mint endpoint returns 501 and we fall back to the same-origin URL with
// a TIGHTER sandbox (no allow-same-origin) — keeping the iframe usable
// for static previews while blocking the cross-frame escalation vector.
// Network errors also fall back so a transient failure doesn't break
// the preview entirely. The fallback banner in the empty/state surface
// tells the user how to restore full capability (enable isolation).
//
// See docs/superpowers/specs/2026-05-27-preview-origin-isolation-design.md.
async function _setPreviewSrc(frame, sameOriginUrl) {
  if (!frame || !sameOriginUrl) return;
  if (!currentCoder().workspaceId) {
    console.log('[coder.preview] set src (no workspace; same-origin)', { url: sameOriginUrl });
    _applySandbox(frame, 'same-origin');
    frame.src = sameOriginUrl;
    return;
  }
  try {
    const tokenStart = performance.now();
    const resp = await fetch(
      `/api/coder/preview-token/${encodeURIComponent(currentCoder().workspaceId)}`,
      { method: 'POST', credentials: 'include' },
    );
    console.log('[coder.preview] token mint response', {
      status: resp.status,
      ok: resp.ok,
      ms: Math.round(performance.now() - tokenStart),
      hint: resp.status === 501 ? 'isolation disabled (expected) — falling back to same-origin' : undefined,
    });
    if (resp.ok) {
      const data = await resp.json();
      if (data && data.token && data.isolated_origin) {
        const sep = sameOriginUrl.includes('?') ? '&' : '?';
        const target = `${data.isolated_origin}${sameOriginUrl}${sep}_pvt=${encodeURIComponent(data.token)}`;
        console.log('[coder.preview] set src (isolated origin)', {
          target, sameOriginUrl, origin: data.isolated_origin,
        });
        _applySandbox(frame, 'isolated');
        frame.src = target;
        return;
      }
    }
    // 501 (isolation off), 404 (workspace gone), 503 (store init failed)
    // → same-origin fallback. Don't surface as an error; the preview
    // still loads, just without origin separation (and without
    // allow-same-origin in the sandbox — see _applySandbox).
  } catch (err) {
    console.log('[coder.preview] token mint network error → same-origin fallback', {
      error: err?.message || String(err),
    });
    // Network error — same-origin fallback.
  }
  console.log('[coder.preview] set src (same-origin fallback)', { url: sameOriginUrl });
  _applySandbox(frame, 'same-origin');
  frame.src = sameOriginUrl;
}

// One-time install of iframe load/error listeners so the dev console
// shows the lifecycle: when the iframe actually starts/finishes loading,
// and when the browser surfaces an error. Idempotent — the previewFrame
// element is long-lived; we attach once and the events fire across all
// src changes (file preview, port preview, expiry re-mint).
let _previewFrameListenersInstalled = false;
function _installPreviewFrameListeners() {
  if (_previewFrameListenersInstalled) return;
  const frame = currentCoder().dom.previewFrame;
  if (!frame) return;
  _previewFrameListenersInstalled = true;
  frame.addEventListener('load', () => {
    // src reads as the resolved URL the browser navigated to. For the
    // about:blank reset between previews this also fires — we tag the
    // mode in dataset so the log makes the distinction obvious.
    console.log('[coder.preview] iframe load event', {
      src: frame.src,
      mode: frame.dataset.previewMode || '(none)',
      filePreview: currentCoder().filePreview ? { path: currentCoder().filePreview.filePath, name: currentCoder().filePreview.fileName } : null,
    });
  });
  frame.addEventListener('error', (e) => {
    // The browser fires `error` on the iframe only for a small subset
    // of network failures (DNS, refused). HTTP 4xx/5xx don't trigger
    // this — the iframe just displays the error body. Combine this log
    // with the server-side preview_file route logs for full picture.
    console.warn('[coder.preview] iframe error event', {
      src: frame.src,
      mode: frame.dataset.previewMode || '(none)',
      message: e?.message || '(no message)',
    });
  });
}

// One-time install: listens for postMessage from the isolated iframe
// when its preview session expires (the proxy returns 401 +
// X-Augmentum-Preview-Expired:true; the bootstrap script the proxy
// injects relays that via postMessage to us). Re-mints and reloads.
//
// Origin validation: only accepts messages from an origin matching the
// iframe's current src origin, so an unrelated tab can't poison this.
let _previewExpiryInstalled = false;
function _installPreviewExpiryListener() {
  if (_previewExpiryInstalled) return;
  _previewExpiryInstalled = true;
  window.addEventListener('message', (event) => {
    const data = event.data;
    if (!data || typeof data !== 'object') return;
    if (data.type !== 'augmentum.preview.expired') return;
    const frame = currentCoder().dom.previewFrame;
    if (!frame) return;
    const currentSrc = frame.getAttribute('src') || '';
    let frameOrigin = '';
    try {
      frameOrigin = new URL(currentSrc, window.location.href).origin;
    } catch (_e) { return; }
    if (!frameOrigin || event.origin !== frameOrigin) return;
    const sameOriginUrl = frame.dataset.previewSameOriginUrl
      || currentCoder().activePreviewUrl;
    if (!sameOriginUrl) return;
    // Blank and re-set so the new token actually takes effect.
    frame.removeAttribute('src');
    _setPreviewSrc(frame, sameOriginUrl);
  });
}

// Relay console/error events captured from the live preview (injected shim →
// postMessage) to the per-workspace beacon, so the coder model sees the errors
// the USER actually hit — the headless browser tools cold-load a fresh page and
// structurally miss them. Same origin-validation as the expiry listener.
let _previewConsoleInstalled = false;
function _installPreviewConsoleListener() {
  if (_previewConsoleInstalled) return;
  _previewConsoleInstalled = true;
  window.addEventListener('message', (event) => {
    const data = event.data;
    if (!data || typeof data !== 'object') return;
    if (data.type !== 'augmentum.preview.console') return;
    if (!Array.isArray(data.entries) || !data.entries.length) return;
    const frame = currentCoder().dom.previewFrame;
    if (!frame) return;
    const currentSrc = frame.getAttribute('src') || '';
    let frameOrigin = '';
    try {
      frameOrigin = new URL(currentSrc, window.location.href).origin;
    } catch (_e) { return; }
    if (!frameOrigin || event.origin !== frameOrigin) return;
    const wsId = currentCoder().workspaceId;
    if (!wsId) return;
    try {
      fetch(`/api/coder/workspaces/${encodeURIComponent(wsId)}/preview-console`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ entries: data.entries.slice(0, 50) }),
        keepalive: true,
      }).catch(() => {});
    } catch (_e) { /* best-effort — never let telemetry break the UI */ }
  });
}

// --- Live-preview capture bridge --------------------------------------------
// When the user's preview is open, the coder captures the frame their real GPU
// already rendered instead of re-rendering a heavy WebGL page headless (which
// is 6-45s+ or times out in the GPU-less workspace). The proxy injects a
// capture agent into the preview; this WS carries capture requests down from
// the server and relays the resulting PNG data URL back up. Server side:
// augmentum/coder/preview_capture.py + /ws/coder/preview-capture/{ws}.
let _captureWs = null;
let _captureWsWorkspace = null;
let _previewCaptureListenerInstalled = false;

function _previewFrameOrigin() {
  const frame = currentCoder().dom.previewFrame;
  if (!frame) return '';
  const src = frame.getAttribute('src') || '';
  try { return new URL(src, window.location.href).origin; } catch (_e) { return ''; }
}

function _closePreviewCaptureWs() {
  if (_captureWs) { try { _captureWs.close(); } catch (_e) { /* already closing */ } }
  _captureWs = null;
  _captureWsWorkspace = null;
}

function _installPreviewCaptureListener() {
  if (_previewCaptureListenerInstalled) return;
  _previewCaptureListenerInstalled = true;
  // Frame -> parent: the injected agent postMessages a captured PNG; relay it
  // to the server over the capture WS. Same origin-validation as the other
  // preview listeners so an unrelated tab can't inject a frame.
  window.addEventListener('message', (event) => {
    const data = event.data;
    if (!data || typeof data !== 'object') return;
    if (data.type !== 'augmentum.preview.capture.result') return;
    const fo = _previewFrameOrigin();
    if (!fo || event.origin !== fo) return;
    if (_captureWs && _captureWs.readyState === 1 /* OPEN */) {
      try {
        _captureWs.send(JSON.stringify({
          type: 'result', id: data.id,
          data_url: data.data_url || '',
          width: data.width || 0, height: data.height || 0,
          reason: data.reason || '',
        }));
      } catch (_e) { /* socket race — server times out and falls back */ }
    }
  });
}

// Open (or keep) the capture WS for the active workspace. Dedups by id and
// closes a stale socket on workspace switch. A falsy id closes it.
function _ensurePreviewCaptureWs(workspaceId) {
  if (!workspaceId) { _closePreviewCaptureWs(); return; }
  if (_captureWs && _captureWsWorkspace === workspaceId
      && (_captureWs.readyState === 0 || _captureWs.readyState === 1)) return;
  _closePreviewCaptureWs();
  _installPreviewCaptureListener();
  _captureWsWorkspace = workspaceId;
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${window.location.host}/ws/coder/preview-capture/${encodeURIComponent(workspaceId)}`;
  let ws;
  try { ws = new WebSocket(url); } catch (_e) { return; }
  _captureWs = ws;
  // Server -> parent -> iframe: forward a capture request into the preview.
  ws.onmessage = (ev) => {
    let msg; try { msg = JSON.parse(ev.data); } catch (_e) { return; }
    if (!msg || msg.type !== 'capture') return;
    const frame = currentCoder().dom.previewFrame;
    const relayFail = (reason) => {
      try { ws.send(JSON.stringify({ type: 'result', id: msg.id, data_url: '', reason })); } catch (_e) {}
    };
    if (!frame || !frame.contentWindow) { relayFail('no-frame'); return; }
    try {
      frame.contentWindow.postMessage(
        { type: 'augmentum.preview.capture', id: msg.id },
        _previewFrameOrigin() || '*',
      );
    } catch (_e) { relayFail('postmessage-failed'); }
  };
  ws.onclose = () => { if (_captureWs === ws) _captureWs = null; };
  ws.onerror = () => { /* onclose handles teardown */ };
}

function _renderPreviewPane() {
  if (!currentCoder().dom.previewPane || !currentCoder().dom.previewEmpty || !currentCoder().dom.previewFrame) return;
  _installPreviewExpiryListener();
  _installPreviewConsoleListener();
  _installPreviewFrameListeners();
  // Keep the live-capture socket bound to the active workspace so
  // browser_screenshot can grab the user's GPU-rendered frame.
  _ensurePreviewCaptureWs(currentCoder().workspaceId);

  // File-preview short-circuit. Renders the file URL in the iframe and
  // bypasses every dev-server code path below — including the auto-kick
  // to terminal when state is 'not_published'. The dev-server poll
  // continues to update currentCoder().previewInfo / currentCoder().previewPorts in the background;
  // the renderer just ignores it as long as a file preview is open.
  if (currentCoder().filePreview && currentCoder().workspaceId) {
    currentCoder().dom.previewPortList?.classList.add('hidden');
    currentCoder().dom.previewState && (currentCoder().dom.previewState.textContent = currentCoder().filePreview.fileName || 'File preview');
    currentCoder().dom.previewUrl && (currentCoder().dom.previewUrl.textContent = currentCoder().filePreview.filePath || '');
    currentCoder().dom.previewEmpty.classList.add('hidden');
    currentCoder().dom.previewFrame.classList.remove('hidden');
    // Visibility-gated lifecycle: never load (or keep loaded) a preview
    // document the user can't see — it would execute on this page's main
    // thread the whole time. Defer with the parked marker instead;
    // _updatePreviewFrameActivity resumes through this same branch.
    if (!_previewFrameVisible()) {
      currentCoder().dom.previewFrame.dataset.previewSameOriginUrl = currentCoder().filePreview.url;
      _parkPreviewFrame(currentCoder().dom.previewFrame);
    } else if (currentCoder().dom.previewFrame.getAttribute('src') !== currentCoder().filePreview.url) {
      delete currentCoder().dom.previewFrame.dataset.previewParked;
      currentCoder().dom.previewFrame.dataset.previewSameOriginUrl = currentCoder().filePreview.url;
      _setPreviewSrc(currentCoder().dom.previewFrame, currentCoder().filePreview.url);
    }
    // Surface the preview tab + reload + close button. The "open
    // external" + "save" buttons stay hidden — the route requires an
    // authenticated session, so handing the user a raw URL would only
    // surface a 401 when they paste it elsewhere.
    currentCoder().dom.workbenchPreviewTab?.classList.remove('hidden');
    currentCoder().dom.previewToggleBtn?.classList.remove('hidden');
    if (currentCoder().dom.previewToggleBtn) currentCoder().dom.previewToggleBtn.textContent = 'Preview';
    currentCoder().dom.previewReloadBtn?.classList.toggle('hidden', currentCoder().activeWorkbenchTab !== 'preview');
    currentCoder().dom.previewOpenExternalBtn?.classList.add('hidden');
    currentCoder().dom.previewSaveBtn?.classList.add('hidden');
    _ensureFilePreviewCloseButton();
    return;
  }
  _hideFilePreviewCloseButton();

  const readyPorts = currentCoder().previewPorts.filter(p => p.listening && p.host_port);
  const urls = (currentCoder().previewInfo.urls || []).filter(Boolean);
  if (urls.length > 0 && (!urls.includes(currentCoder().activePreviewUrl))) {
    currentCoder().activePreviewUrl = currentCoder().previewInfo.primary_url || urls[0] || '';
  }
  if (!currentCoder().dom.previewPortList) return;

  const hasChoices = readyPorts.length > 1;
  currentCoder().dom.previewPortList.classList.toggle('hidden', !hasChoices);
  currentCoder().dom.previewPortList.innerHTML = hasChoices
    ? readyPorts.map((p) => {
      // Proxy through Augmentum so the iframe stays same-origin and the
      // dev server is reachable from any device that can reach Augmentum
      // (phone-on-LAN, tablet, etc.). Mirrors _preview_proxy_path() in
      // augmentum/proxy/coder_routes.py.
      const url = `/api/coder/preview/${encodeURIComponent(currentCoder().workspaceId)}/${p.container_port}/`;
      const active = url === currentCoder().activePreviewUrl ? ' active' : '';
      return `<button class="coder-preview-port-btn${active}" data-preview-url="${escapeHtml(url)}" title="Open container port ${p.container_port} in preview">:${p.container_port}</button>`;
    }).join('')
    : '';
  currentCoder().dom.previewPortList.querySelectorAll('[data-preview-url]').forEach((btn) => {
    btn.addEventListener('click', () => {
      currentCoder().activePreviewUrl = btn.dataset.previewUrl || '';
      _renderPreviewPane();
    });
  });

  if (currentCoder().previewInfo.state === 'ready' && currentCoder().activePreviewUrl) {
    currentCoder().dom.previewState.textContent = `${currentCoder().previewInfo.ready_count || readyPorts.length} live`;
    currentCoder().dom.previewUrl.textContent = currentCoder().activePreviewUrl;
    currentCoder().dom.previewEmpty.classList.add('hidden');
    currentCoder().dom.previewFrame.classList.remove('hidden');
    // Compare via getAttribute, NOT iframe.src. The DOM property normalizes
    // to an absolute URL ('http://host/api/...') while currentCoder().activePreviewUrl is
    // relative ('/api/...'), so `iframe.src !== currentCoder().activePreviewUrl` was true
    // every poll cycle — re-assigning src reloads the iframe, which is the
    // 5-second flash users see. getAttribute returns the original string.
    // Visibility-gated lifecycle: a dev server turning ready while the
    // user is on the Terminal tab (the common case — the agent just
    // started it) must NOT load the app in the background, and the 5s
    // ports poll must not reload a parked one. Defer with the parked
    // marker; _updatePreviewFrameActivity resumes through this branch.
    if (!_previewFrameVisible()) {
      currentCoder().dom.previewFrame.dataset.previewSameOriginUrl = currentCoder().activePreviewUrl;
      _parkPreviewFrame(currentCoder().dom.previewFrame);
    } else if (currentCoder().dom.previewFrame.getAttribute('src') !== currentCoder().activePreviewUrl) {
      delete currentCoder().dom.previewFrame.dataset.previewParked;
      // Set the same-origin URL marker so the isolated-mode handler
      // can recover it on expiry / re-mint without losing the trail.
      currentCoder().dom.previewFrame.dataset.previewSameOriginUrl = currentCoder().activePreviewUrl;
      // Fire-and-forget — _setPreviewSrc mints a token if isolation
      // is enabled and falls back to same-origin on 501 (off).
      _setPreviewSrc(currentCoder().dom.previewFrame, currentCoder().activePreviewUrl);
    }
  } else {
    currentCoder().dom.previewFrame.classList.add('hidden');
    if (currentCoder().dom.previewFrame.getAttribute('src')
        && currentCoder().dom.previewFrame.getAttribute('src') !== 'about:blank') {
      currentCoder().dom.previewFrame.src = 'about:blank';
    }
    if (currentCoder().previewInfo.state === 'published_idle') {
      currentCoder().dom.previewState.textContent = 'Waiting for app';
      currentCoder().dom.previewUrl.textContent = 'Start a dev server on a common port';
      currentCoder().dom.previewEmpty.textContent = 'Ports are exposed for this workspace. Start a dev server on 3000, 5173, 8000, 8080, or another published port to load it here.';
    } else {
      currentCoder().dom.previewState.textContent = 'Preview unavailable';
      currentCoder().dom.previewUrl.textContent = 'No preview selected';
      currentCoder().dom.previewEmpty.textContent = 'Enable preview for this workspace to open a local app here.';
    }
    currentCoder().dom.previewEmpty.classList.remove('hidden');
  }

  currentCoder().dom.previewToggleBtn?.classList.toggle('hidden', currentCoder().previewInfo.state === 'not_published');
  if (currentCoder().dom.previewToggleBtn) {
    currentCoder().dom.previewToggleBtn.textContent = currentCoder().previewInfo.state === 'ready' ? 'Preview' : 'Preview status';
  }
  currentCoder().dom.workbenchPreviewTab?.classList.toggle('hidden', currentCoder().previewInfo.state === 'not_published');
  if (currentCoder().dom.workbenchPreviewTab) {
    currentCoder().dom.workbenchPreviewTab.textContent = currentCoder().previewInfo.state === 'ready' ? 'Preview' : 'Preview';
  }
  // Mirror onto the top pane-switch's Preview button (only meaningful once
  // a preview exists to switch to).
  currentCoder().dom.paneSwitch?.querySelector('.coder-pane-btn[data-pane="preview"]')
    ?.classList.toggle('hidden', currentCoder().previewInfo.state === 'not_published');
  currentCoder().dom.previewReloadBtn?.classList.toggle('hidden', currentCoder().activeWorkbenchTab !== 'preview' || !currentCoder().previewInfo.ready);
  currentCoder().dom.previewOpenExternalBtn?.classList.toggle('hidden', currentCoder().activeWorkbenchTab !== 'preview' || !currentCoder().previewInfo.ready);
  currentCoder().dom.previewSaveBtn?.classList.toggle('hidden', currentCoder().activeWorkbenchTab !== 'preview' || !currentCoder().previewInfo.ready);

  if (currentCoder().previewInfo.state === 'not_published' && currentCoder().activeWorkbenchTab === 'preview') {
    // If a file preview is somehow still set here, that's a logic bug —
    // the short-circuit at the top should have returned before reaching
    // this branch. Log loudly so a regression is obvious instead of
    // silently kicking the user back to terminal mid-interaction.
    if (currentCoder().filePreview) {
      console.warn(
        '[coder.preview] auto-kick to terminal fired while file preview is active — short-circuit bypassed (bug)',
        { filePreview: currentCoder().filePreview, previewInfo: currentCoder().previewInfo },
      );
    } else {
      console.log('[coder.preview] auto-kick to terminal (no preview state)');
    }
    _setWorkbenchTab('terminal');
  }
}

function _openPreview(url = '') {
  if (currentCoder().previewInfo.state === 'not_published') return;
  if (url) currentCoder().activePreviewUrl = url;
  else if (!currentCoder().activePreviewUrl) currentCoder().activePreviewUrl = currentCoder().previewInfo.primary_url || (currentCoder().previewInfo.urls || [])[0] || '';
  _showMobileWorkbenchSurface();
  _setWorkbenchTab('preview');
  _renderPreviewPane();
}

/**
 * Open a static file in the preview pane via /api/coder/preview-file.
 *
 * Lives in its own state slot (currentCoder().filePreview) so the 5s dev-server poll
 * (_refreshPorts) can't overwrite it. Renderer logic in
 * _renderPreviewPane checks currentCoder().filePreview first and skips the
 * dev-server state machine entirely when set — including the
 * "not_published → kick back to terminal" branch that was making the
 * pane disappear mid-interaction.
 */
function _openFilePreview(workspaceId, filePath, fileName) {
  if (!workspaceId || !filePath) return;
  const url = buildPreviewUrl(workspaceId, filePath);
  if (!url) return;
  // Use console.log not console.debug — debug is hidden behind the
  // Verbose filter in Chrome DevTools and was making the lifecycle
  // invisible to users diagnosing preview issues.
  console.log('[coder.preview] open file preview', {
    workspaceId, filePath, fileName, url,
  });
  currentCoder().filePreview = { url, filePath, fileName };
  currentCoder().activePreviewUrl = url;
  _showMobileWorkbenchSurface();
  _setWorkbenchTab('preview');
  _renderPreviewPane();
}

// Lazy-created × button atop the preview header for closing the file
// preview and returning the pane to the dev-server view. Created once,
// reused; visibility tracked via the .hidden class so _renderPreviewPane
// can flip it without rebuilding DOM.
let _filePreviewCloseBtnEl = null;

function _ensureFilePreviewCloseButton() {
  if (_filePreviewCloseBtnEl) {
    _filePreviewCloseBtnEl.classList.remove('hidden');
    return;
  }
  // Mount alongside the existing preview action buttons. previewReloadBtn
  // is the closest reliably-mounted sibling — we drop the × right next to it.
  const host = currentCoder().dom.previewReloadBtn?.parentElement;
  if (!host) return;
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'icon-btn small coder-file-preview-close';
  btn.title = 'Close file preview';
  btn.setAttribute('aria-label', 'Close file preview');
  btn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
  btn.addEventListener('click', () => _closeFilePreview());
  host.appendChild(btn);
  _filePreviewCloseBtnEl = btn;
}

function _hideFilePreviewCloseButton() {
  _filePreviewCloseBtnEl?.classList.add('hidden');
}

/**
 * Close the file preview and return the pane to whatever the dev-server
 * state machine wants (port chip / "no preview" empty state). Called
 * from the × button rendered atop the preview header when a file
 * preview is active.
 */
function _closeFilePreview() {
  if (!currentCoder().filePreview) return;
  console.log('[coder.preview] close file preview', {
    filePath: currentCoder().filePreview.filePath, fileName: currentCoder().filePreview.fileName,
  });
  currentCoder().filePreview = null;
  // Force the iframe back to about:blank before _renderPreviewPane
  // decides what to show — otherwise the old file URL lingers for a
  // frame while the dev-server state evaluates.
  if (currentCoder().dom.previewFrame) currentCoder().dom.previewFrame.src = 'about:blank';
  currentCoder().activePreviewUrl = currentCoder().previewInfo.primary_url || (currentCoder().previewInfo.urls || [])[0] || '';
  _renderPreviewPane();
}

function _resetPreviewState() {
  currentCoder().previewInfo = {
    state: 'not_published',
    published: false,
    ready: false,
    ready_count: 0,
    primary_url: null,
    urls: [],
  };
  currentCoder().previewPorts = [];
  currentCoder().activePreviewUrl = '';
  currentCoder().filePreview = null;
  if (currentCoder().dom.previewPortList) currentCoder().dom.previewPortList.innerHTML = '';
  _renderPreviewPane();
}

/**
 * Refresh the MCP status tile in the coder status bar with the current
 * server and tool counts. Best-effort — silently hides on network error
 * so the status bar doesn't fill with warnings when MCP is disabled.
 */
async function _refreshMcpTile() {
  if (!currentCoder().dom.mcpTile || !currentCoder().dom.mcpLabel) return;
  try {
    const resp = await fetch('/v1/mcp/servers?health=false');
    if (!resp.ok) {
      currentCoder().dom.mcpTile.classList.add('hidden');
      return;
    }
    const data = await resp.json();
    if (!data.enabled) {
      currentCoder().dom.mcpTile.classList.add('hidden');
      return;
    }
    const servers = data.servers || [];
    const toolCount = servers.reduce((n, s) => n + (s.tool_count || 0), 0);
    if (toolCount === 0) {
      currentCoder().dom.mcpTile.classList.add('hidden');
      return;
    }
    currentCoder().dom.mcpLabel.textContent = `${toolCount} MCP tool${toolCount !== 1 ? 's' : ''} via ${servers.length} server${servers.length !== 1 ? 's' : ''}`;
    currentCoder().dom.mcpTile.classList.remove('hidden');
  } catch {
    currentCoder().dom.mcpTile.classList.add('hidden');
  }
}

async function _refreshPowerTile() {
  if (!currentCoder().dom.powerTile || !currentCoder().dom.powerLabel) return;
  if (currentCoder().runtimePowerActivation?.display_name) {
    const prefix = currentCoder().runtimePowerActivation?.source === 'controller' ? 'Auto Power' : 'Power';
    currentCoder().dom.powerLabel.textContent = `${prefix}: ${currentCoder().runtimePowerActivation.display_name}`;
    currentCoder().dom.powerTile.classList.remove('hidden');
    return;
  }
  try {
    const mod = await import('./powers.js');
    await mod.refreshCoderPowerTile(currentCoder().dom.powerTile, currentCoder().dom.powerLabel, currentCoder().workspaceId || '');
  } catch {
    currentCoder().dom.powerTile.classList.add('hidden');
  }
}

function _setRuntimePowerActivation(payload) {
  currentCoder().runtimePowerActivation = payload || null;
  _refreshPowerTile();
}

/**
 * Initialize Coder mode. Called when user switches to Coder mode.
 */
export async function init() {
  if (_initialized) return;

  currentCoder().dom.terminalPane = document.getElementById('coder-terminal-pane');
  currentCoder().dom.terminalStack = document.getElementById('coder-terminal-stack');
  currentCoder().dom.editorPane = document.getElementById('coder-editor-pane');
  currentCoder().dom.fileTree = document.getElementById('coder-file-tree');
  currentCoder().dom.filesTitle = document.getElementById('coder-files-title');
  currentCoder().dom.workspaceSelect = document.getElementById('coder-workspace-select');
  currentCoder().dom.workspaceBar = document.getElementById('coder-workspace-bar');
  currentCoder().dom.editorSplit = document.getElementById('coder-editor-split');
  currentCoder().dom.editorTabs = document.getElementById('coder-editor-tabs');
  currentCoder().dom.statusEl = document.getElementById('coder-status');
  currentCoder().dom.statusText = document.getElementById('coder-status-text');
  currentCoder().dom.statusDetail = document.getElementById('coder-status-detail');
  _ensureRunDetailsButton();
  currentCoder().dom.intentBar = document.getElementById('coder-intent');
  currentCoder().dom.workbenchTerminalTab = document.getElementById('coder-workbench-terminal-tab');
  currentCoder().dom.workbenchPreviewTab = document.getElementById('coder-workbench-preview-tab');
  currentCoder().dom.previewPane = document.getElementById('coder-preview-pane');
  currentCoder().dom.previewFrame = document.getElementById('coder-preview-frame');
  currentCoder().dom.previewEmpty = document.getElementById('coder-preview-empty');
  currentCoder().dom.previewState = document.getElementById('coder-preview-state');
  currentCoder().dom.previewUrl = document.getElementById('coder-preview-url');
  currentCoder().dom.previewPortList = document.getElementById('coder-preview-port-list');
  currentCoder().dom.previewReloadBtn = document.getElementById('coder-preview-reload-btn');
  currentCoder().dom.previewOpenExternalBtn = document.getElementById('coder-preview-open-external-btn');
  currentCoder().dom.previewSaveBtn = document.getElementById('coder-preview-save-btn');
  currentCoder().dom.previewSavePrompt = document.getElementById('coder-save-prompt');
  currentCoder().dom.previewToggleBtn = document.getElementById('coder-preview-toggle-btn');
  currentCoder().dom.mcpTile = document.getElementById('coder-mcp-tile');
  currentCoder().dom.mcpLabel = document.getElementById('coder-mcp-label');
  currentCoder().dom.powerTile = document.getElementById('coder-power-tile');
  currentCoder().dom.powerLabel = document.getElementById('coder-power-label');
  currentCoder().dom.packsTile = document.getElementById('coder-packs-tile');
  currentCoder().dom.workbenchTerminalTab?.addEventListener('click', () => _setWorkbenchTab('terminal'));
  currentCoder().dom.workbenchPreviewTab?.addEventListener('click', () => _openPreview());
  currentCoder().dom.previewToggleBtn?.addEventListener('click', () => _openPreview());
  currentCoder().dom.previewReloadBtn?.addEventListener('click', () => {
    const frame = currentCoder().dom.previewFrame;
    if (!frame) return;
    // Re-apply the sandbox so a reload always reflects the CURRENT trust
    // state (and re-syncs the trust banner via _refreshPreviewTrustUi).
    // Without this, "refresh the preview" never re-evaluated trust, so a
    // just-trusted workspace could keep showing the sandbox banner.
    _applySandbox(frame, frame.dataset.previewMode || 'same-origin');
    // contentWindow.location access throws in tight mode (origin null), so
    // guard it and fall back to re-navigating the iframe src.
    try { frame.contentWindow?.location.reload(); } catch {
      frame.src = currentCoder().activePreviewUrl || currentCoder().previewInfo.primary_url || 'about:blank';
    }
  });
  currentCoder().dom.previewOpenExternalBtn?.addEventListener('click', () => {
    const url = currentCoder().activePreviewUrl || currentCoder().previewInfo.primary_url;
    if (url) window.open(url, '_blank', 'noopener');
  });
  currentCoder().dom.previewSaveBtn?.addEventListener('click', async () => {
    if (!currentCoder().workspaceId) return;
    try {
      const { openSavePrompt } = await import('./coder-library-save.js');
      await openSavePrompt(
        currentCoder().workspaceId,
        currentCoder().dom.previewSaveBtn,
        currentCoder().dom.previewSavePrompt,
      );
    } catch (exc) {
      // Lazy-import failure is rare but surface it so the user isn't
      // left with a silently-broken button.
      showToast('Save prompt failed to load', 'error');
    }
  });
  if (currentCoder().dom.mcpTile) {
    currentCoder().dom.mcpTile.addEventListener('click', async () => {
      // Open settings and switch to Providers → MCP. `openSettings` is
      // imported from settings.js at module scope in app.js but not
      // available here directly — dynamic import keeps this file
      // dependency-free and still opens the right pane.
      try {
        const { openSettings } = await import('./settings.js');
        await openSettings();
        // Switch to the providers tab + mcp subtab.
        const modal = document.getElementById('settings-modal');
        modal?.querySelector('[data-tab="providers"]')?.click();
        modal?.querySelector('[data-prov-tab="mcp"]')?.click();
      } catch { /* ignore */ }
    });
    _refreshMcpTile();
    // Refresh every 30s while the coder page is mounted. Skip when the tab
    // is backgrounded — like the other coder polls — so a hidden tab never
    // wakes the main thread for a tile repaint it can't be seen.
    if (!window._coderMcpTileTimer) {
      window._coderMcpTileTimer = setInterval(() => {
        if (!document.hidden) _refreshMcpTile();
      }, 30000);
    }
  }
  if (currentCoder().dom.packsTile) {
    currentCoder().dom.packsTile.addEventListener('click', async () => {
      // Quick access to offline docs packs: Settings → Knowledge with
      // the curated "Coder" catalog shelf pre-selected. The category
      // pills render async after the catalog fetch, so poll briefly
      // for the pill before clicking it — deep-link, not a new surface.
      try {
        const { openSettings } = await import('./settings.js');
        await openSettings();
        const modal = document.getElementById('settings-modal');
        const knowledgeNav = modal?.querySelector('[data-tab="knowledge"]');
        // Knowledge is an admin-only pane (packs are server-level).
        if (!knowledgeNav || knowledgeNav.offsetParent === null) {
          showToast('Knowledge packs are managed by the server admin', 'info');
          return;
        }
        knowledgeNav.click();
        const started = Date.now();
        const tryPill = () => {
          const pill = modal?.querySelector('#knowledge-category-pills .knowledge-pill[data-category="Coder"]');
          if (pill) { if (!pill.classList.contains('active')) pill.click(); return; }
          if (Date.now() - started < 4000) setTimeout(tryPill, 200);
        };
        tryPill();
      } catch { /* ignore */ }
    });
  }
  if (currentCoder().dom.powerTile) {
    currentCoder().dom.powerTile.addEventListener('click', async () => {
      try {
        const { openSettings } = await import('./settings.js');
        await openSettings();
        const modal = document.getElementById('settings-modal');
        modal?.querySelector('[data-tab="automation"]')?.click();
        modal?.querySelector('[data-auto-tab="powers"]')?.click();
      } catch { /* ignore */ }
    });
    _refreshPowerTile();
    if (!window._coderPowerTileTimer) {
      window._coderPowerTileTimer = setInterval(() => {
        if (!document.hidden) _refreshPowerTile();
      }, 30000);
    }
  }
  currentCoder().dom.mobileTabs = document.getElementById('coder-mobile-tabs');
  currentCoder().dom.coderLayout = document.getElementById('coder-layout');
  currentCoder().dom.paneSwitch = document.getElementById('coder-pane-switch');
  currentCoder().dom.paneSwitch?.querySelectorAll('.coder-pane-btn, .coder-pane-action').forEach((btn) => {
    btn.addEventListener('click', () => {
      const action = btn.dataset.paneAction;
      if (action === 'files') {
        // Toggle the left-panel (file tree) drawer.
        const leftPanel = document.querySelector('.left-panel');
        const backdrop = document.querySelector('.panel-backdrop');
        if (leftPanel) {
          const isOpen = leftPanel.classList.contains('open');
          leftPanel.classList.toggle('open', !isOpen);
          if (backdrop) backdrop.classList.toggle('visible', !isOpen);
        }
      } else if (action === 'exit') {
        app.setMode('passthrough');
      } else if (btn.dataset.pane) {
        _setCoderPane(btn.dataset.pane);
      }
    });
  });

  document.getElementById('coder-new-file-btn')?.addEventListener('click', () => _beginInlineCreate(currentCoder().workspaceId, '/workspace', false));
  document.getElementById('coder-new-folder-btn')?.addEventListener('click', () => _beginInlineCreate(currentCoder().workspaceId, '/workspace', true));
  document.getElementById('coder-refresh-tree-btn')?.addEventListener('click', () => {
    if (currentCoder().workspaceId) _populateFileTree(currentCoder().workspaceId);
  });
  document.getElementById('coder-search-toggle-btn')?.addEventListener('click', () => {
    if (!currentCoder().workspaceId) { showToast('No workspace selected', 'warning'); return; }
    toggleCoderSearch();
  });
  initCoderSearch({
    getWorkspaceId: () => currentCoder().workspaceId || '',
    openResult: (hit) => _openFileInEditor(
      currentCoder().workspaceId, hit.path, hit.name,
      { line: hit.line, spans: hit.spans },
    ),
    buildIndex: async () => {
      if (!currentCoder().workspaceId) return false;
      try {
        await fetch(`/api/coder/index/${encodeURIComponent(currentCoder().workspaceId)}`, { method: 'POST' });
        _startIndexProgressPoll(currentCoder().workspaceId);
        return true;
      } catch { return false; }
    },
  });
  _initUploadAndDrop();
  document.getElementById('coder-export-workspace-btn')?.addEventListener('click', _downloadWorkspace);
  document.getElementById('coder-port-expose-btn')?.addEventListener('click', _publishPortsForActiveWorkspace);

  document.getElementById('coder-workspace-select')?.addEventListener('change', (e) => {
    _switchWorkspace(e.target.value);
  });

  document.getElementById('coder-add-workspace-btn')?.addEventListener('click', () => {
    _openNewWorkspaceModal();
  });

  document.getElementById('coder-manage-workspaces-btn')?.addEventListener('click', () => {
    _openWorkspaceManagerModal();
  });

  document.getElementById('coder-safeguards-btn')?.addEventListener('click', async () => {
    if (!currentCoder().workspaceId) { showToast('No workspace selected', 'error'); return; }
    const next = !currentCoder().safeguardsEnabled;
    if (!next && !confirm(
      'Disable safeguards for this workspace?\n\n' +
      'Soft circuit-breakers (action stagnation, test-failure streak, ' +
      'inspection nudges, read-repeat refusal, etc.) will stop firing. ' +
      'Use only for strong API-backed or strong local models that ' +
      'legitimately run long — weaker models will likely loop.'
    )) return;
    try {
      const resp = await fetch(
        `/api/coder/workspaces/${encodeURIComponent(currentCoder().workspaceId)}/safeguards`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled: next }),
        },
      );
      if (!resp.ok) { showToast('Toggle failed', 'error'); return; }
      currentCoder().safeguardsEnabled = next;
      _updateSafeguardsButton();
      showToast(next ? 'Safeguards enabled' : 'Safeguards disabled', 'success');
    } catch { showToast('Toggle failed', 'error'); }
  });

  document.getElementById('coder-always-on-btn')?.addEventListener('click', async () => {
    if (!currentCoder().workspaceId) { showToast('No workspace selected', 'error'); return; }
    const next = !currentCoder().alwaysOn;
    try {
      const resp = await fetch(
        `/api/coder/workspaces/${encodeURIComponent(currentCoder().workspaceId)}/always-on`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ always_on: next }),
        },
      );
      if (!resp.ok) { showToast('Toggle failed', 'error'); return; }
      currentCoder().alwaysOn = next;
      _updateAlwaysOnButton();
      showToast(
        next
          ? 'Always-on enabled (container won’t auto-stop)'
          : 'On-demand (container will auto-stop when idle)',
        'success',
      );
    } catch { showToast('Toggle failed', 'error'); }
  });

  document.getElementById('coder-lan-btn')?.addEventListener('click', async () => {
    if (!currentCoder().workspaceId) { showToast('No workspace selected', 'error'); return; }
    const next = !currentCoder().lanAccessible;
    const action = next
      ? 'Make this workspace\'s ports reachable from your LAN? The container will be briefly recreated (files are safe).'
      : 'Restrict ports to localhost only? The container will be briefly recreated (files are safe).';
    if (!confirm(action)) return;
    const btn = document.getElementById('coder-lan-btn');
    if (btn) { btn.disabled = true; btn.title = 'Toggling…'; }
    try {
      const resp = await fetch(
        `/api/coder/workspaces/${encodeURIComponent(currentCoder().workspaceId)}/lan`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled: next }),
        },
      );
      if (!resp.ok) { showToast('Toggle failed', 'error'); return; }
      currentCoder().lanAccessible = next;
      _updateLanButton();
      await _reconnectActiveTerminal();
      _startGitPolling();
      await _refreshPorts();
      showToast(
        next
          ? 'LAN access enabled — services are reachable from your network'
          : 'LAN access disabled — ports are loopback-only',
        'success',
      );
    } catch { showToast('Toggle failed', 'error'); }
    finally { if (btn) btn.disabled = false; }
  });

  document.getElementById('coder-stop-workspace-btn')?.addEventListener('click', async () => {
    if (!currentCoder().workspaceId) { showToast('No workspace selected', 'error'); return; }
    // No destructive-confirm modal — stopping is non-destructive
    // (DB row + volume survive, restart is one chat away). Match the
    // tap-fast UX of an editor's pause button.
    try {
      const resp = await fetch(
        `/api/coder/workspaces/${encodeURIComponent(currentCoder().workspaceId)}/stop`,
        { method: 'POST' },
      );
      if (!resp.ok) { showToast('Stop failed', 'error'); return; }
      currentCoder().status = 'stopped';
      _updateLifecycleButtons();
      showToast('Workspace stopped', 'success');
    } catch { showToast('Stop failed', 'error'); }
  });

  document.getElementById('coder-start-workspace-btn')?.addEventListener('click', async () => {
    if (!currentCoder().workspaceId) { showToast('No workspace selected', 'error'); return; }
    const wasPaused = (currentCoder().status === 'paused');
    try {
      const resp = await fetch(
        `/api/coder/workspaces/${encodeURIComponent(currentCoder().workspaceId)}/start`,
        { method: 'POST' },
      );
      if (!resp.ok) {
        // Surface the "container has been removed" error from our
        // reconcile path so the user knows to recreate. Other 5xx
        // get a generic message.
        let detail = '';
        try { detail = (await resp.json())?.error || ''; } catch (_) {}
        if (detail.toLowerCase().includes('recreate')) {
          showToast('Container is gone — delete this workspace and create a new one', 'error', 8000);
        } else {
          showToast(detail || 'Start failed', 'error');
        }
        return;
      }
      currentCoder().status = 'running';
      _updateLifecycleButtons();
      showToast(wasPaused ? 'Workspace resumed' : 'Workspace started', 'success');
    } catch { showToast('Start failed', 'error'); }
  });

  // Heavyweight model (per-workspace). Dual-purpose slot — backs Bug
  // Finder verification AND stagnation-escalation in the coder loop.
  // Empty value → both roles fall back: verifier uses primary, and
  // escalation is disabled (stuck turns surface the standard error
  // pill instead of auto-handing off).
  //
  // Uses the same shared model picker as the chat composer
  // (``openModelPickerFor`` in settings.js) — search, recent models,
  // backend grouping, vision badges, Manage button. Same UX everywhere
  // a model is chosen.
  document.getElementById('coder-verifier-btn')?.addEventListener('click', async () => {
    if (!currentCoder().workspaceId) { showToast('No workspace selected', 'error'); return; }
    const anchor = document.getElementById('coder-verifier-btn');
    if (!anchor) return;
    const { openModelPickerFor } = await import('./settings.js');
    await openModelPickerFor({
      anchor,
      currentValue: _activeVerifierModel,
      onSelect: async (name) => {
        await _saveWorkspaceHeavyweightModel(name);
      },
    });
  });
  // Right-click → clear the override. The picker itself doesn't have
  // a "(none)" item today, so we wire the most discoverable alternative:
  // right-click matches the rest of the chrome's affordance pattern
  // (Tooltip mentions it when set).
  document.getElementById('coder-verifier-btn')?.addEventListener('contextmenu', async (e) => {
    e.preventDefault();
    if (!currentCoder().workspaceId) return;
    if (!_activeVerifierModel) return;   // nothing to clear
    if (!window.confirm(
      `Clear heavyweight model override for this workspace?\n\n` +
      `Current: ${_activeVerifierModel}\n\n` +
      `Bug Finder will fall back to the primary, and stagnation ` +
      `auto-escalation will be disabled.`,
    )) return;
    await _saveWorkspaceHeavyweightModel('');
  });

  document.getElementById('coder-delete-workspace-btn')?.addEventListener('click', async () => {
    if (!currentCoder().workspaceId) { showToast('No workspace selected', 'error'); return; }
    const select = document.getElementById('coder-workspace-select');
    const wsName = select?.selectedOptions[0]?.textContent || currentCoder().workspaceId;
    const choice = await _confirmWorkspaceRemoval(wsName);
    if (!choice) return;
    const purge = choice.purge;
    try {
      const url = purge
        ? `/api/coder/workspaces/${encodeURIComponent(currentCoder().workspaceId)}?purge=1`
        : `/api/coder/workspaces/${encodeURIComponent(currentCoder().workspaceId)}`;
      const resp = await fetch(url, { method: 'DELETE' });
      if (!resp.ok) { showToast(purge ? 'Remove failed' : 'Archive failed', 'error'); return; }
      // Tear down terminal and editor
      if (currentCoder().terminalId) { Terminal.destroy(currentCoder().terminalId); currentCoder().terminalId = null; }
      if (currentCoder().activeEditorId) { Editor.destroy(currentCoder().activeEditorId); currentCoder().activeEditorId = null; }
      _stopGitPolling();
      currentCoder().workspaceId = null;
      _persistActiveWorkspaceId(null);
      currentCoder().activeFilePath = '';
      currentCoder().chatHistory = [];
      currentCoder().conversation?.clear();
      currentCoder().missionPanel?.clear();
      currentCoder().editorFiles = [];
      currentCoder().activeEditorFile = null;
      // Refresh workspace list
      await _populateWorkspaceSelect();
      // Re-enter coder mode to pick next workspace or show empty state
      _onEnterCoderMode();
      showToast(purge ? 'Workspace removed' : 'Workspace archived', 'success');
    } catch { showToast(purge ? 'Remove failed' : 'Archive failed', 'error'); }
  });

  // Commit panel — opens the stage/diff/commit flow. Distinct from
  // the push button so users can review what they're committing before
  // it touches the remote.
  document.getElementById('coder-git-commit-btn')?.addEventListener('click', () => {
    if (!currentCoder().workspaceId) { showToast('No workspace selected', 'warning'); return; }
    _openCommitPanel();
  });

  // Branch chip — click opens the branch picker popover (switch + create).
  document.getElementById('coder-git-branch')?.addEventListener('click', (ev) => {
    if (!currentCoder().workspaceId) return;
    ev.stopPropagation();
    _toggleBranchPopover(ev.currentTarget);
  });

  // Git push
  document.getElementById('coder-git-push-btn')?.addEventListener('click', async () => {
    if (!currentCoder().workspaceId) return;
    showToast('Pushing...', 'info', 3000);
    try {
      const resp = await fetch(`/api/coder/workspaces/${currentCoder().workspaceId}/git/push`, { method: 'POST' });
      const data = await resp.json();
      if (resp.ok) { showToast('Pushed successfully', 'success'); _refreshGitHeaderStatus(); }
      else {
        showToast(extractErrorMessage(data, 'Push failed'), 'error');
        if (resp.status === 401) _openGitSettingsModal();
      }
    } catch (err) { showToast('Push failed: ' + err.message, 'error'); }
  });

  // Git pull. Backend classifies the common failure shapes (auth /
  // dirty_tree / non_fast_forward / no_upstream / network) so we can
  // route the user to the right next step instead of dumping the raw
  // git error.
  document.getElementById('coder-git-pull-btn')?.addEventListener('click', async () => {
    if (!currentCoder().workspaceId) return;
    showToast('Pulling...', 'info', 3000);
    try {
      const resp = await fetch(`/api/coder/workspaces/${currentCoder().workspaceId}/git/pull`, { method: 'POST' });
      const data = await resp.json();
      if (resp.ok) {
        showToast('Pulled successfully', 'success');
        _refreshGitHeaderStatus();
        _populateFileTree(currentCoder().workspaceId);
        return;
      }
      const msg = data.error || 'Pull failed';
      const code = data.error_code || '';
      // 401 → auth: bounce the user into the settings modal directly.
      if (resp.status === 401 || code === 'auth') {
        showToast(msg, 'error', 6000);
        _openGitSettingsModal();
        return;
      }
      // 409 → known recoverable shape. Surface a longer toast so the
      // explanation has time to read; tree refresh stays in case the
      // working tree drifted during the attempt.
      if (resp.status === 409) {
        showToast(msg, 'warning', 8000);
        _refreshGitHeaderStatus();
        return;
      }
      showToast(msg, 'error');
    } catch (err) { showToast('Pull failed: ' + err.message, 'error'); }
  });

  // Git settings
  document.getElementById('coder-git-settings-btn')?.addEventListener('click', _openGitSettingsModal);

  // Checkpoints status-bar tile → reveal the checkpoints section.
  // Left panel may be collapsed (desktop) or drawer-closed (mobile);
  // expose it either way, then expand the list and scroll to it.
  document.getElementById('coder-checkpoint-tile')?.addEventListener('click', () => {
    const leftPanel = document.querySelector('.left-panel');
    const appEl = document.getElementById('app');
    if (leftPanel) {
      if (window.matchMedia('(min-width: 768px)').matches) {
        leftPanel.classList.remove('desktop-collapsed');
        if (appEl) appEl.setAttribute('data-panel', 'visible');
      } else {
        leftPanel.classList.add('open');
        document.querySelector('.panel-backdrop')?.classList.add('visible');
      }
    }
    // Make sure the Coder Files view is the active panel tab.
    document.getElementById('coder-files-view')?.classList.remove('hidden');
    // Expand the checkpoints accordion + kick a refresh, then scroll.
    const toggle = document.getElementById('coder-checkpoints-toggle');
    const list = document.getElementById('coder-checkpoints-list');
    if (toggle && list && !_checkpointsExpanded) {
      _checkpointsExpanded = true;
      list.style.display = '';
      toggle.classList.add('expanded');
      _loadCheckpoints();
    }
    requestAnimationFrame(() => {
      document.getElementById('coder-checkpoints')
        ?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
  });

  // Mobile-only close button for the full-screen editor overlay.
  // Closes all editor tabs and slides the pane out.
  document.getElementById('coder-editor-close-mobile')?.addEventListener('click', () => {
    currentCoder().editorFiles = [];
    currentCoder().activeEditorFile = null;
    if (currentCoder().activeEditorId) { Editor.destroy(currentCoder().activeEditorId); currentCoder().activeEditorId = null; }
    if (currentCoder().dom.editorTabs) currentCoder().dom.editorTabs.innerHTML = '';
    _hideEditorSplit();
  });

  // Listen for // agent requests from terminal
  document.addEventListener('coder:agent-request', (e) => {
    const { request } = e.detail;
    if (request && currentCoder().workspaceId) {
      // Route to conversation if not in classic mode
      const layout = document.getElementById('coder-layout');
      if (currentCoder().conversation && layout && !layout.classList.contains('classic-mode')) {
        _runAgentInConversation(request);
      } else {
        _runAgentInTerminal(request);
      }
    }
  });

  // Listen for Escape cancel from terminal during agent execution
  document.addEventListener('coder:agent-cancel', () => {
    // Cancel conversation stream if active
    if (currentCoder().coderStream?.isActive()) {
      _stopActiveCoderRun('user_cancel');
      currentCoder().conversation?.addError('Cancelled by user');
      _updateStatus('idle', 'cancelled');
      _updateIntentBar();
    }
    if (currentCoder().terminalAgentAbort) {
      currentCoder().terminalAgentAbort.abort();
      if (currentCoder().terminalId) {
        Terminal.write(currentCoder().terminalId, '\r\n\x1b[33m[Agent cancelled]\x1b[0m\r\n');
        Terminal.setAgentActive(currentCoder().terminalId, false);
      }
    }
  });

  // Global cancel hotkeys — Ctrl+C / Esc at document level, so the
  // user can stop a runaway agent from ANY focus (chat input, side
  // panel, nav) not just the terminal. Previously only the terminal
  // had these bound (via xterm's custom key handler), so users typing
  // a follow-up in the chat box while the agent thrashed had no way
  // to interrupt without clicking into the terminal first. Gated on
  // "agent is actually running" so we don't hijack normal copy on
  // idle pages. Input fields are skipped for ^C specifically so copy
  // still works when the user has text selected in an editor;
  // Escape always triggers because it's not a copy key. The chat
  // textarea needs an explicit check because browsers let ^C fire
  // keydown without an active selection.
  document.addEventListener('keydown', (e) => {
    const agentActive =
      currentCoder().coderStream?.isActive() ||
      (currentCoder().terminalAgentAbort && !currentCoder().terminalAgentAbort.signal.aborted);
    if (!agentActive) return;
    const isEsc = e.key === 'Escape';
    const isCtrlC = (e.ctrlKey || e.metaKey) && (e.key === 'c' || e.key === 'C');
    if (!isEsc && !isCtrlC) return;
    // For Ctrl+C, only intercept when there is NO active text
    // selection — copying selected text stays a higher-priority
    // action than cancel. Escape always wins because it has no
    // overlapping browser default.
    if (isCtrlC) {
      const sel = window.getSelection?.();
      if (sel && sel.toString().length > 0) return;
    }
    e.preventDefault();
    e.stopPropagation();
    document.dispatchEvent(new CustomEvent('coder:agent-cancel'));
  }, true); // capture phase so we beat editor handlers

  // Listen for mode changes. Double-RAF on entry mirrors the initial-
  // page-load path (see the end of init()): applyMode() unhides the
  // terminal pane synchronously, but the coder-layout grid hasn't
  // reflowed yet, so a Terminal.create() called this tick measures
  // clientWidth/clientHeight = 0 and opens xterm into a 0×0 canvas.
  // Waiting two frames guarantees at least one paint cycle has run.
  document.addEventListener('augmentum:mode-changed', (e) => {
    if (e.detail.mode === 'coder') {
      requestAnimationFrame(() => requestAnimationFrame(() => {
        _onEnterCoderMode();
      }));
    } else {
      _onLeaveCoderMode();
    }
  });

  // Terminal gave up reconnecting (container likely stopped). Clear
  // the cached terminal id so the next _onEnterCoderMode() rebuilds
  // from scratch — without this, the stale id blocks re-creation and
  // the user sees a dead canvas until they recreate the workspace.
  document.addEventListener('coder:terminal-disconnected', () => {
    if (currentCoder().terminalId) {
      try { Terminal.destroy(currentCoder().terminalId); } catch {}
      currentCoder().terminalId = null;
    }
  });

  // Save conversation on page unload. sendBeacon can silently drop the
  // request if the browser refuses it (payload too large, permission, or
  // unsupported); fall back to keepalive fetch so the conversation isn't
  // lost on refresh.
  window.addEventListener('beforeunload', () => {
    if (!currentCoder().workspaceId || !currentCoder().conversation) return;
    const url = `/api/coder/conversation/${encodeURIComponent(currentCoder().workspaceId)}`;
    const payload = JSON.stringify({ messages: currentCoder().conversation.getHistory() });
    try {
      const sent = navigator.sendBeacon(
        url, new Blob([payload], { type: 'application/json' }),
      );
      if (sent) return;
    } catch { /* sendBeacon unavailable or refused */ }
    try {
      fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
        keepalive: true,
      }).catch(() => {});
    } catch { /* best-effort */ }
  });

  // Refit terminal on viewport changes (orientation, keyboard, etc.)
  // rAF-coalesce so a drag-resize burst doesn't trigger one xterm.fit per
  // resize event (each fit recalculates cell metrics — expensive enough
  // to stall the compositor when stacked with other handlers).
  let _refitPending = false;
  const _refitTerminal = () => {
    if (_refitPending) return;
    _refitPending = true;
    requestAnimationFrame(() => {
      _refitPending = false;
      if (currentCoder().terminalId) Terminal.fit(currentCoder().terminalId);
    });
  };
  window.addEventListener('resize', _refitTerminal);
  if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', _refitTerminal);
  }

  // Listen for theme changes to update terminal
  document.addEventListener('augmentum:theme-changed', () => {
    Terminal.updateTheme();
  });

  // Refresh workspace select immediately when a terminal connects — attaching
  // to a previously "exited" container means it just came back up.
  document.addEventListener('coder:terminal-connected', () => {
    _populateWorkspaceSelect();
  });

  // Listen for input from extra keys bar
  document.addEventListener('coder:send-input', (e) => {
    if (currentCoder().terminalId) {
      Terminal.sendInput(currentCoder().terminalId, e.detail.data);
    }
  });

  // Exit coder mode button
  document.getElementById('coder-exit-btn')?.addEventListener('click', () => {
    app.setMode('passthrough');
  });

  // Coder theme cycle button. Lives inside the checkpoints header — stop
  // propagation so clicking the theme doesn't toggle the checkpoints list.
  document.getElementById('coder-theme-cycle-btn')?.addEventListener('click', (e) => {
    e.stopPropagation();
    _cycleCoderTheme();
  });

  // Classic mode toggle
  document.getElementById('coder-classic-toggle')?.addEventListener('click', () => {
    const layout = document.getElementById('coder-layout');
    if (!layout) return;
    const isClassic = layout.classList.toggle('classic-mode');
    localStorage.setItem('augmentum-coder-classic', isClassic ? '1' : '0');
    // Re-fit terminal after layout change
    if (currentCoder().terminalId) requestAnimationFrame(() => Terminal.fit(currentCoder().terminalId));
    showToast(isClassic ? 'Classic mode (terminal only)' : 'Conversation mode', 'success');
  });

  // Conversation input — send on Enter (Shift+Enter for newline)
  const coderInput = document.getElementById('coder-input');
  const coderSendBtn = document.getElementById('coder-send-btn');

  coderInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      // Cooperative chord (2026-05-31 — default flipped to steer):
      //   Enter (plain)   → Send. While idle → new turn. While
      //                     running → Steer (drains at next iteration
      //                     boundary, redirects the in-flight turn).
      //                     Previously this was end-of-turn Queue but
      //                     that caused intent drift — by the time
      //                     the queued message landed the situation
      //                     had moved on.
      //   Ctrl/Cmd+Enter  → Queue (drains at end-of-turn and chains
      //                     as a brand-new turn). Use when you have
      //                     a follow-up that should wait for the
      //                     current work to finish cleanly instead
      //                     of redirecting it.
      // Shift+Enter stays as newline-in-textarea (handled by the
      // default behavior — we never reach this branch when shift
      // is down).
      const mode = (e.ctrlKey || e.metaKey) ? 'queue' : 'auto';
      _sendConversationMessage({ mode });
    }
  });
  // Auto-resize textarea via the shared rAF-deferred helper. The 150px
  // cap is tighter than the chat composer's 200px because the coder
  // pane is dense with terminal + conversation panels — a tall input
  // crowds the workbench. See ``utils/textarea-autosize.js`` for the
  // full rationale on why the inline pattern thrashes layout.
  coderInput?.addEventListener('input', () => {
    scheduleAutosize(coderInput, 150);
  });
  coderSendBtn?.addEventListener('click', _sendConversationMessage);

  // Rewind last turn — opens a 3-mode menu (both / files / conv).
  // Backend cancels any in-flight turn first; client also aborts
  // the local stream so the UI unwinds immediately.
  const coderRewindBtn = document.getElementById('coder-rewind-btn');
  coderRewindBtn?.addEventListener('click', _showRewindMenu);

  // Pause / Resume — visible only while a run is active.
  const coderPauseBtn = document.getElementById('coder-pause-btn');
  coderPauseBtn?.addEventListener('click', _togglePauseActiveRun);

  // Planning-mode cycle (default → plan → auto → default). Clickable
  // on the badge OR Shift+Tab in the textarea.
  const coderPlanModeBtn = document.getElementById('coder-plan-mode-btn');
  coderPlanModeBtn?.addEventListener('click', _cyclePlanningMode);

  // Thinking toggle — per-turn enable_thinking override for capable
  // models. Mirrors the chat composer's thinking-toggle pattern.
  // Visibility + state are driven by the current model + workspace.
  const coderThinkingBtn = document.getElementById('coder-thinking-btn');
  coderThinkingBtn?.addEventListener('click', _toggleCoderThinking);
  // Refresh visibility on input focus — catches the case where the
  // user changed models in the picker after coder mount. Cheap; just
  // a capability check + a class flip.
  coderInput?.addEventListener('focus', _refreshCoderThinkingToggle);
  // Initial render — show or hide based on the model active at load.
  _refreshCoderThinkingToggle();

  // Shift+Tab in the textarea cycles planning mode without taking
  // focus off the input — matches the CC convention. preventDefault
  // because the default Shift+Tab moves focus to the previous element.
  coderInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Tab' && e.shiftKey && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();
      _cyclePlanningMode();
    }
  });

  // Paste-to-attach. Browsers expose clipboard files via
  // ``e.clipboardData.items`` with kind==='file'. A screenshot pasted
  // straight from Cmd/Ctrl+Shift+S ends up here too, auto-named
  // image.png by the browser. We preventDefault ONLY when files are
  // actually present — plain text paste still inserts normally.
  coderInput?.addEventListener('paste', async (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    const files = [];
    for (const item of items) {
      if (item.kind === 'file') {
        const f = item.getAsFile();
        if (f) files.push(f);
      }
    }
    if (!files.length) return;  // text paste — let it through
    e.preventDefault();
    await _ingestAttachments(files);
  });

  // Drag-and-drop attachments onto the conversation pane. Images go
  // through the chat_images pipeline (base64 → VL models see the image
  // directly). Non-image files upload into /workspace/.augmentum/
  // attachments/ and the outgoing message gets a "📎 Attached: <path>"
  // footer so the agent can file_read them.
  const convEl = document.getElementById('coder-conversation');
  if (convEl) {
    let convDragDepth = 0;
    const convClearActive = () => {
      convDragDepth = 0;
      convEl.classList.remove('coder-conversation--drop-active');
    };
    convEl.addEventListener('dragenter', (e) => {
      if (!e.dataTransfer?.types?.includes('Files')) return;
      e.preventDefault();
      convDragDepth += 1;
      convEl.classList.add('coder-conversation--drop-active');
    });
    convEl.addEventListener('dragover', (e) => {
      if (!e.dataTransfer?.types?.includes('Files')) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    });
    convEl.addEventListener('dragleave', () => {
      convDragDepth -= 1;
      if (convDragDepth <= 0) convClearActive();
    });
    convEl.addEventListener('drop', async (e) => {
      if (!e.dataTransfer?.types?.includes('Files')) return;
      e.preventDefault();
      convClearActive();
      const files = Array.from(e.dataTransfer?.files || []);
      if (!files.length) return;
      await _ingestAttachments(files);
    });
  }

  // Onboarding chips — dispatch to agent
  document.getElementById('coder-conv-messages')?.addEventListener('click', (e) => {
    const chip = e.target.closest('.coder-conv-chip');
    if (!chip?.dataset.prompt) return;
    const input = document.getElementById('coder-input');
    if (input) {
      input.value = chip.dataset.prompt;
      _sendConversationMessage();
    }
  });

  // Ctrl+S / Cmd+S to save in editor
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 's' && currentCoder().activeEditorId) {
      e.preventDefault();
      _saveFile(currentCoder().workspaceId, currentCoder().activeFilePath);
    }
  });

  // Ctrl+` to focus terminal (like VS Code)
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === '`') {
      e.preventDefault();
      if (currentCoder().terminalId) Terminal.focus(currentCoder().terminalId);
    }
  });

  // Ctrl+Shift+F opens workspace search (VS Code parity). Only when the
  // coder files panel is actually mounted + a workspace is active, so it
  // doesn't shadow the shortcut in other modes.
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === 'F' || e.key === 'f')) {
      if (!currentCoder().workspaceId) return;
      const panel = document.getElementById('coder-files-view');
      if (!panel || panel.classList.contains('hidden')) return;
      e.preventDefault();
      openCoderSearch();
    }
  });

  // Escape closes the most transient coder surface first.
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (currentCoder().dom.editorSplit && !currentCoder().dom.editorSplit.classList.contains('hidden')) {
      _hideEditorSplit();
      return;
    }
    if (_selectedPaths.size) { _clearSelection(); return; }
    if (currentCoder().activeWorkbenchTab === 'preview' && currentCoder().previewInfo.state !== 'not_published') {
      _setWorkbenchTab('terminal');
    }
  });

  // Intent bar — click handler
  currentCoder().dom.intentBar?.addEventListener('click', (e) => {
    const btn = e.target.closest('.coder-intent-btn');
    if (!btn || !btn.dataset.action) return;
    const action = btn.dataset.action;
    if (action === '__cancel') {
      // Cancel conversation stream
      if (currentCoder().coderStream?.isActive()) {
        _stopActiveCoderRun('user_cancel');
        currentCoder().conversation?.addError('Cancelled by user');
      }
      // Cancel running agent or active prompt (terminal path)
      if (currentCoder().activePromptDisposable) {
        currentCoder().activePromptDisposable.dispose();
        currentCoder().activePromptDisposable = null;
        if (currentCoder().terminalId) {
          Terminal.write(currentCoder().terminalId, '\r\n\x1b[33m[Cancelled]\x1b[0m\r\n');
          Terminal.setAgentActive(currentCoder().terminalId, false);
        }
      } else if (currentCoder().terminalAgentAbort) {
        currentCoder().terminalAgentAbort.abort();
        if (currentCoder().terminalId) {
          Terminal.write(currentCoder().terminalId, '\r\n\x1b[33m[Cancelled]\x1b[0m\r\n');
          Terminal.setAgentActive(currentCoder().terminalId, false);
        }
      }
      _updateStatus('idle', 'cancelled');
      _updateIntentBar();
    } else if (action.startsWith('?:')) {
      // Expansion loop — prompt user for details before sending to agent
      _promptAndRun(action);
    } else if (action.startsWith('//')) {
      _runAgentInTerminal(action.slice(2));
    } else if (action.startsWith('\x03')) {
      // Ctrl+C — send to terminal
      if (currentCoder().terminalId) Terminal.sendInput(currentCoder().terminalId, '\x03');
    } else {
      // Shell command
      if (currentCoder().terminalId) {
        Terminal.sendInput(currentCoder().terminalId, action + '\n');
      }
    }
  });

  // Debounced terminal output listener to update intents
  let _intentDebounce = null;
  document.addEventListener('coder:terminal-output', () => {
    clearTimeout(_intentDebounce);
    _intentDebounce = setTimeout(_updateIntentBar, 800);
  });

  // Mobile bottom tabs
  currentCoder().dom.mobileTabs?.querySelectorAll('.coder-mob-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.coderTab;

      if (tab === 'chat' || tab === 'terminal') {
        // Route through the unified controller so the top pane-switch and
        // this emoji bar stay in sync (and the layout/extra-keys logic
        // lives in one place).
        _setCoderPane(tab);
      } else if (tab === 'files') {
        // Toggle the left panel drawer on mobile
        const leftPanel = document.querySelector('.left-panel');
        const backdrop = document.querySelector('.panel-backdrop');
        if (leftPanel) {
          const isOpen = leftPanel.classList.contains('open');
          leftPanel.classList.toggle('open', !isOpen);
          if (backdrop) backdrop.classList.toggle('visible', !isOpen);
        }
      } else if (tab === 'exit') {
        app.setMode('passthrough');
      }
    });
  });

  // Keyboard inset tracking removed: interactive-widget=resizes-content
  // in the viewport meta makes the ICB shrink natively when the keyboard
  // opens, so 100dvh and fixed bottom:0 already track the keyboard at
  // compositor speed. The old JS sync (visualViewport → --coder-keyboard-
  // inset CSS var) lagged the keyboard animation by 1+ frames.

  _initialized = true;

  // Park/resume the preview iframe when the whole browser tab is
  // backgrounded/foregrounded — a hidden tab's iframe JS keeps running
  // (timers keep firing) and shares the coder UI's main thread. See
  // _updatePreviewFrameActivity for the full rationale.
  document.addEventListener('visibilitychange', _updatePreviewFrameActivity);

  // Deep-link target for coder.run.* notifications ("Open workspace" on a
  // finished background mission). Window-global by the same convention as
  // window.openCompanionNotes — the notifications surface must not import
  // coder.js (it loads on every page; coder is lazy).
  window.openCoderWorkspace = (workspaceId) => {
    try {
      if (app.state?.mode !== 'coder') app.setMode('coder');
    } catch (err) {
      console.warn('openCoderWorkspace mode switch failed', err);
    }
    if (workspaceId && workspaceId !== currentCoder().workspaceId) {
      // _switchWorkspace's generation counter supersedes the mode-entry
      // default-workspace load racing alongside this call.
      _switchWorkspace(workspaceId).catch((err) => {
        console.warn('openCoderWorkspace switch failed', err);
        showToast('Could not open that workspace', 'error');
      });
    }
  };
  // Open the new-workspace modal from anywhere (companion "New workspace"
  // pick). Switches into coder mode first, then opens the create modal with
  // the template/repo choices intact — the user still picks those (never
  // auto-select). A suggested name derived from the build prompt pre-fills the
  // name field; the prompt itself is stashed for the create flow to seed as
  // the first task once workspace-seeding tasks land.
  window.openCoderNewWorkspace = (opts) => {
    const o = opts || {};
    try {
      if (app.state?.mode !== 'coder') app.setMode('coder');
    } catch (err) {
      console.warn('openCoderNewWorkspace mode switch failed', err);
    }
    if (o.prompt) {
      try { window.__coderPendingSeedPrompt = String(o.prompt); } catch (_) { /* best-effort */ }
    }
    Promise.resolve(_openNewWorkspaceModal()).then(() => {
      const nameEl = document.getElementById('coder-nw-name');
      const suggested = String(o.suggested_name || '').trim();
      if (nameEl && suggested) nameEl.value = suggested;
    }).catch((err) => {
      console.warn('openCoderNewWorkspace modal failed', err);
    });
  };

  // The notifications surface fires this synchronous DOM event on action
  // click (banner AND toast) before POSTing the action — user-gesture
  // context for navigation. Server handler for coder.run.* just acks.
  window.addEventListener('augmentum:notification-action', (ev) => {
    const n = ev?.detail?.notification;
    const actionId = ev?.detail?.actionId;
    if (!n || actionId !== 'open') return;
    if (!String(n.channel_id || '').startsWith('coder.run.')) return;
    const ws = n.payload?.workspace_id;
    if (ws) window.openCoderWorkspace(ws);
  });

  // Warm the tooling-profile cache so the workspace creation modal
  // dropdown renders from server truth (catalog lives in
  // augmentum/coder/profiles.py). Fire-and-forget — the fallback list
  // covers the brief window before this resolves.
  _loadToolingProfiles().catch(() => {});

  // Command palette — universal launcher (Ctrl/Cmd+K). Initialized
  // here so it can use coder.js's workspace switching + file opener
  // without command-palette.js having to import them statically (which
  // would also pull coder.js's heavy deps into any other surface that
  // wants the palette). The registry pattern means new actions land
  // by calling registerCommand() — no edits to the palette itself.
  try {
    const { initCommandPalette, registerCommand } = await import('./command-palette.js');
    const ready = initCommandPalette({
      getActiveWorkspaceId: () => currentCoder().workspaceId,
      openFile: (wsId, path, name) => {
        if (!wsId) return;
        _openFileInEditor(wsId, path, name);
      },
      switchWorkspace: (wsId) => {
        if (!wsId) return;
        openWorkspaceById(wsId).catch(() => { /* surfaced via toast */ });
      },
    });
    if (ready) _registerCoderCommands(registerCommand);
  } catch (err) {
    console.warn('Command palette init failed', err);
  }

  // If we're already in coder mode (page reload), enter it.
  // Defer to allow the browser to complete layout reflow — on reload,
  // applyMode() removes the hidden class from the terminal pane but the
  // grid hasn't reflowed yet, so the container has 0×0 dimensions.
  // Double-RAF guarantees at least one paint cycle has completed.
  const currentMode = document.getElementById('app')?.getAttribute('data-mode');
  if (currentMode === 'coder') {
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    await _onEnterCoderMode();
  }
}

async function _onEnterCoderMode() {
  _applyCoderTheme();
  _ensureExtraKeys();

  // Reconcile extra-keys visibility with the currently active mobile
  // tab. Default on first load is Chat, where the terminal-specific
  // Ctrl/Esc/arrows are useless clutter and the 48px reserve wastes
  // space. If the user was on Terminal before leaving coder mode,
  // respect that on re-entry.
  if (window.innerWidth < 768) {
    const activeBtn = document.querySelector('.coder-mob-tab.active');
    const activeTab = activeBtn?.dataset.coderTab || 'chat';
    const showExtraKeys = (activeTab === 'terminal');
    const extraKeys = document.getElementById('coder-extra-keys');
    const mainArea = document.querySelector('.main-area');
    if (extraKeys) extraKeys.style.display = showExtraKeys ? '' : 'none';
    if (mainArea) mainArea.classList.toggle('coder-has-extra-keys', showExtraKeys);
  }

  // Collapse the left panel on first entry per page load so coder mode opens
  // uncluttered. Users can reopen via the header menu button; thereafter we
  // leave their choice alone. We modify DOM state directly rather than calling
  // closePanel() so we don't overwrite the user's global panel preference.
  if (!_panelAutoClosed) {
    _panelAutoClosed = true;
    try {
      const leftPanel = document.querySelector('.left-panel');
      const appEl = document.getElementById('app');
      if (leftPanel && window.matchMedia('(min-width: 768px)').matches) {
        leftPanel.classList.add('desktop-collapsed');
        if (appEl) appEl.setAttribute('data-panel', 'hidden');
      }
    } catch { /* ignore */ }
  }
  // Show extra keys bar (may have been hidden by _onLeaveCoderMode)
  const extraKeys = document.getElementById('coder-extra-keys');
  if (extraKeys) extraKeys.style.display = '';

  // ── Reparent terminal into layout grid ────────────────────────────
  // The terminal pane starts in the main-area as a direct child.
  // Move it into the terminal-wrapper inside the coder-layout grid.
  const termStack = document.getElementById('coder-terminal-stack');
  if (currentCoder().dom.terminalPane && termStack && !termStack.contains(currentCoder().dom.terminalPane)) {
    termStack.appendChild(currentCoder().dom.terminalPane);
    currentCoder().dom.terminalPane.classList.remove('hidden');
  }

  // ── Rehome surface-adopted elements back into main-area ───────────
  // Defensive: CoderSurface.mount() used to adopt the editor split, the
  // status bar, and the intent bar into .surface-content. #surface-grid
  // is display:none in coder mode (coder.css:55), so anything parked
  // there renders as a 0x0 box — classes toggle correctly but nothing
  // appears. The editor split is owned by #coder-layout; the status and
  // intent bars are direct children of .main-area. Pull each back to
  // its rightful HTML home if a stale session parked it under .surface-
  // content. New mounts skip the adoption entirely (see coder-surface.js),
  // so this only runs on the first coder-mode entry after upgrade.
  const coderLayout = document.getElementById('coder-layout');
  const mainAreaEl = document.querySelector('.main-area');
  if (currentCoder().dom.editorSplit && coderLayout && !coderLayout.contains(currentCoder().dom.editorSplit)) {
    coderLayout.appendChild(currentCoder().dom.editorSplit);
  }
  if (currentCoder().dom.statusEl && mainAreaEl && currentCoder().dom.statusEl.parentElement !== mainAreaEl) {
    mainAreaEl.appendChild(currentCoder().dom.statusEl);
    currentCoder().dom.statusEl.classList.remove('hidden');
  }
  if (currentCoder().dom.intentBar && mainAreaEl && currentCoder().dom.intentBar.parentElement !== mainAreaEl) {
    mainAreaEl.appendChild(currentCoder().dom.intentBar);
    currentCoder().dom.intentBar.classList.remove('hidden');
  }

  // Restore classic mode preference
  const layout = document.getElementById('coder-layout');
  if (layout && localStorage.getItem('augmentum-coder-classic') === '1') {
    layout.classList.add('classic-mode');
  }
  _setWorkbenchTab(currentCoder().activeWorkbenchTab || 'terminal');
  _renderPreviewPane();

  // ── Initialize conversation component ─────────────────────────────
  if (!currentCoder().conversation) {
    const convMessages = document.getElementById('coder-conv-messages');
    const convScroll = document.getElementById('coder-conv-scroll');
    if (convMessages && convScroll) {
      currentCoder().conversation = new CoderConversation(convMessages, convScroll);
    }
  }

  // ── Initialize mission panel ──────────────────────────────────────
  if (!currentCoder().missionPanel) {
    const panelEl = document.getElementById('coder-mission-panel');
    if (panelEl) currentCoder().missionPanel = new MissionPanel(panelEl);
  }

  // Preload the previewable-file-types registry so right-click → Preview
  // is enabled by the time the user opens any file-tree context menu.
  // Fire-and-forget: failure leaves the Preview item hidden, never
  // breaks the menu's other actions.
  loadPreviewableExtensions();

  // Kick off xterm CDN load IN PARALLEL with the workspace fetch below.
  // Terminal.load() does ~4 dynamic imports + a CSS link load (~50-200ms
  // first time, no-op after); blocking on it serially would push every
  // downstream await one Terminal-load further. The promise is awaited
  // right before Terminal.create(); by then it's almost always settled.
  const _xtermLoadPromise = Terminal.load().catch(() => {});

  // ── Resolve workspace ─────────────────────────────────────────────
  // Fetch once, thread the result through populateSelect + metadata
  // bind below. Prior code issued 3-4 identical /api/coder/workspaces
  // calls per mode-entry; each one runs Docker containers.list under
  // the hood and adds ~200ms of round-trip latency.
  let workspaces = await _fetchWorkspaces();

  // Retry policy: only bother retrying if we *expect* the list to be
  // non-empty — i.e. localStorage remembers a workspace from a prior
  // session, so an empty result is more likely a still-booting server
  // than a genuinely new user. Exponential backoff caps worst case
  // around 1.1s (vs. the prior fixed-1.5s × 2 = 3s wait).
  if (workspaces.length === 0 && _recallActiveWorkspaceId()) {
    for (const delay of [300, 800]) {
      await new Promise(r => setTimeout(r, delay));
      workspaces = await _fetchWorkspaces();
      if (workspaces.length > 0) break;
    }
  }

  // Render the picker with whatever we have (possibly empty) and start
  // the 10s status poll so the chip stays fresh.
  _populateWorkspaceSelect(workspaces);
  _startWorkspaceStatusPolling();

  if (!currentCoder().workspaceId && workspaces.length > 0) {
    // Prefer the workspace the user was actually using before refresh.
    // Falling back to "first running" silently re-binds chat + terminal
    // + file tree to a DIFFERENT workspace when the user has multiple
    // running — which the user reports as "chat got disconnected from
    // terminal". The recall is only honoured when the workspace is
    // still in the running set (stopped containers can't be terminal'd
    // into anyway, and a deleted workspace's id is just stale data).
    const recalled = _recallActiveWorkspaceId();
    const recalledRunning = recalled
      ? workspaces.find(w => w.id === recalled && w.status === 'running')
      : null;
    const running = recalledRunning || workspaces.find(w => w.status === 'running');
    if (running) {
      currentCoder().workspaceId = running.id;
      _persistActiveWorkspaceId(running.id);
      // Re-render the dropdown so its visual `selected` option matches
      // the workspace we just bound to. Without this, the dropdown
      // renders with the FIRST option visually selected (because we
      // populated it earlier when currentCoder().workspaceId was null), so the
      // user sees workspace-A in the picker but the terminal + chat
      // are actually wired to workspace-B. That's the visual half of
      // the "chat/terminal feel out of sync" symptom.
      _populateWorkspaceSelect(workspaces);
    } else {
      // All workspaces are stopped/lost — offer to recreate
      _showStoppedState(workspaces[0]);
      return;
    }
  }

  // ── Bind workspace metadata + kick off parallel fetches ───────────
  // Everything below until the terminal-create block is either pure
  // DOM work or fire-and-forget API calls. Doing them BEFORE the
  // terminal-create await means file tree, conversation, checkpoints,
  // and the codebase index all start fetching while xterm is still
  // doing its WS handshake — instead of after.
  if (currentCoder().workspaceId) {
    if (currentCoder().dom.filesTitle) {
      // Reuse the workspaces list fetched at the top of this function —
      // saves a redundant /api/coder/workspaces round-trip. Falls back
      // to a fresh fetch only if we entered here via a code path that
      // bypassed the upstream fetch (legacy direct calls).
      const wsList = (workspaces && workspaces.length)
        ? workspaces
        : await _fetchWorkspaces();
      const ws = wsList.find(w => w.id === currentCoder().workspaceId);
      if (ws) {
        currentCoder().dom.filesTitle.textContent = ws.name;
        currentCoder().safeguardsEnabled = ws.safeguards_enabled !== false;
        _updateSafeguardsButton();
        _activeVerifierModel = ws.bug_finder_verifier_model || '';
        _updateVerifierButton();
        currentCoder().alwaysOn = !!ws.always_on;
        _updateAlwaysOnButton();
        currentCoder().lanAccessible = !!ws.lan_accessible;
        _updateLanButton();
        currentCoder().status = ws.status || 'stopped';
        _updateLifecycleButtons();
        _updateBugFinderTile(ws);
        _updateAgentsTile(ws);
        if (ws.kind === 'bug_finder') _maybeOpenBugFinderForWorkspace(ws);
      }
    }
    _populateFileTree(currentCoder().workspaceId);
    _startGitPolling();
    if (currentCoder().conversation) _loadConversation(currentCoder().workspaceId);
    // Background: build/update codebase index for semantic search.
    fetch(`/api/coder/index/${encodeURIComponent(currentCoder().workspaceId)}`, { method: 'POST' })
      .then(r => r.json())
      .then(stats => {
        if (stats.total_chunks > 0) {
          console.debug(`[Coder] Index: ${stats.indexed} files indexed, ${stats.total_chunks} chunks (${stats.duration_ms}ms)`);
        }
      })
      .catch(() => {}); // Non-critical — agent works without index
    // Surface the build's progress in the Files panel.
    _startIndexProgressPoll(currentCoder().workspaceId);
  }

  // Initialize checkpoints. Load even when collapsed so the status-bar
  // tile shows an accurate count — the list's display is toggled by
  // _checkpointsExpanded independently. Wired before terminal create
  // so the checkpoints poll can race the WS handshake.
  _initCheckpoints();
  if (currentCoder().workspaceId) _loadCheckpoints();

  // Services panel — collapsible drawer below checkpoints. Init once;
  // load + start polling on every entry so the section reflects the
  // active workspace's registered services.
  _initServices();
  if (currentCoder().workspaceId) {
    _loadServices();
    _startServicesPolling();
  }

  // Wire the preview trust banner so the user can flip same-origin
  // sandbox state for a workspace whose dev server (Vite HMR, etc.)
  // needs cookies/CORS that the tight sandbox kills. Idempotent.
  _initPreviewTrustControls();
  _initChimeControl();

  // ── Create terminal ───────────────────────────────────────────────
  // Awaits xterm CDN load (kicked off in parallel near the top, almost
  // always already settled by now) then opens a WS to the container.
  if (currentCoder().workspaceId && !currentCoder().terminalId && currentCoder().dom.terminalPane) {
    try {
      await _xtermLoadPromise;
      currentCoder().terminalId = await Terminal.create(currentCoder().dom.terminalPane, currentCoder().workspaceId);
      Terminal.focus(currentCoder().terminalId);
    } catch (err) {
      console.error('[Coder] Terminal init failed:', err);
      _showErrorState('Terminal failed to load', err.message);
      return;
    }
  } else if (!currentCoder().workspaceId && currentCoder().dom.terminalPane) {
    // No workspace yet — show create prompt
    _showEmptyState();
  }

  // Set initial status and intent bar
  _updateStatus('idle');
  _updateIntentBar();
}

function _showEmptyState() {
  if (!currentCoder().dom.terminalPane || currentCoder().dom.terminalPane.querySelector('.coder-empty-state')) return;

  const empty = document.createElement('div');
  empty.className = 'coder-empty-state';
  empty.innerHTML = `
    <div class="coder-empty-icon">⚡</div>
    <h3>Welcome to Coder Mode</h3>
    <p>Create a workspace to start coding with AI assistance.</p>
    <div class="coder-empty-actions">
      <button class="coder-create-btn" id="coder-create-workspace-btn">
        New Workspace
      </button>
      <button class="coder-create-btn coder-clone-btn" id="coder-clone-workspace-btn">
        Clone Repository
      </button>
    </div>
    <div class="coder-clone-form hidden" id="coder-clone-form">
      <input type="text" class="field-input" id="coder-clone-url"
        placeholder="https://github.com/user/repo.git">
      <div style="display:flex;gap:var(--space-xs)">
        <input type="text" class="field-input" id="coder-clone-branch"
          placeholder="Branch (default: main)" style="flex:1">
        <input type="text" class="field-input" id="coder-clone-name"
          placeholder="Workspace name" style="flex:1">
      </div>
      <select class="field-select" id="coder-clone-tooling-profile" title="Tooling profile">
        ${_toolingProfileOptions('browser')}
      </select>
      <label class="coder-publish-ports-row" title="Publish common dev-server ports (3000, 5173, 8000, etc.) to 127.0.0.1 so dev servers running inside the workspace are reachable from your browser.">
        <input type="checkbox" id="coder-clone-publish-ports">
        <span>Expose dev-server ports</span>
      </label>
      <button class="coder-create-btn" id="coder-clone-go-btn">Clone &amp; Open</button>
    </div>
  `;
  currentCoder().dom.terminalPane.appendChild(empty);

  empty.querySelector('#coder-create-workspace-btn')?.addEventListener('click', () => {
    _openNewWorkspaceModal();
  });

  empty.querySelector('#coder-clone-workspace-btn')?.addEventListener('click', () => {
    const form = empty.querySelector('#coder-clone-form');
    if (form) form.classList.toggle('hidden');
  });

  empty.querySelector('#coder-clone-go-btn')?.addEventListener('click', async () => {
    const url = currentCoder().dom.terminalPane.querySelector('#coder-clone-url')?.value?.trim();
    if (!url) { showToast('Enter a repository URL', 'warning'); return; }
    const branch = currentCoder().dom.terminalPane.querySelector('#coder-clone-branch')?.value?.trim() || null;
    const nameInput = currentCoder().dom.terminalPane.querySelector('#coder-clone-name')?.value?.trim();
    const publishPorts = !!currentCoder().dom.terminalPane.querySelector('#coder-clone-publish-ports')?.checked;
    const toolingProfile = currentCoder().dom.terminalPane.querySelector('#coder-clone-tooling-profile')?.value || 'browser';
    const repoName = nameInput || url.split('/').pop()?.replace('.git', '') || 'repo';
    showToast(`Cloning ${repoName}...`, 'info', 5000);
    await createWorkspace(repoName, {
      git_url: url,
      git_branch: branch,
      publish_ports: publishPorts,
      tooling_profile: toolingProfile,
    });
  });
}

function _showStoppedState(workspace) {
  if (!currentCoder().dom.terminalPane) return;
  currentCoder().dom.terminalPane.innerHTML = '';

  const stopped = document.createElement('div');
  stopped.className = 'coder-empty-state';
  stopped.innerHTML = `
    <div class="coder-empty-icon">&#x23F8;</div>
    <h3>Workspace stopped</h3>
    <p>"${escapeHtml(workspace.name)}" is no longer running. The container was removed or stopped.</p>
    <button class="coder-create-btn" id="coder-recreate-btn">Create New Workspace</button>
    <button class="coder-create-btn" style="background:none;border:1px solid var(--border);color:var(--text-secondary);margin-top:var(--space-xs)" id="coder-delete-stale-btn">Remove stale workspace</button>
  `;
  currentCoder().dom.terminalPane.appendChild(stopped);

  stopped.querySelector('#coder-recreate-btn')?.addEventListener('click', async () => {
    stopped.querySelector('#coder-recreate-btn').textContent = 'Creating...';
    stopped.querySelector('#coder-recreate-btn').disabled = true;
    // Delete the stale record first
    try { await fetch(`/api/coder/workspaces/${encodeURIComponent(workspace.id)}`, { method: 'DELETE' }); } catch {}
    const result = await createWorkspace(workspace.name || 'home', {
      tooling_profile: workspace.tooling_profile || 'browser',
    });
    if (!result || result.error) {
      stopped.querySelector('#coder-recreate-btn').textContent = 'Failed — try again';
      stopped.querySelector('#coder-recreate-btn').disabled = false;
    }
  });

  stopped.querySelector('#coder-delete-stale-btn')?.addEventListener('click', async () => {
    try { await fetch(`/api/coder/workspaces/${encodeURIComponent(workspace.id)}`, { method: 'DELETE' }); } catch {}
    currentCoder().chatHistory = [];
    currentCoder().conversation?.clear();
    currentCoder().missionPanel?.clear();
    currentCoder().workspaceId = null;
    _showEmptyState();
  });
}

function _showErrorState(title, detail) {
  if (!currentCoder().dom.terminalPane) return;
  currentCoder().dom.terminalPane.innerHTML = '';
  const el = document.createElement('div');
  el.className = 'coder-empty-state';
  el.innerHTML = `
    <div class="coder-empty-icon" style="color:var(--error)">⚠</div>
    <h3>${escapeHtml(title)}</h3>
    <p style="color:var(--text-muted);font-size:var(--text-sm)">${escapeHtml(detail || '')}</p>
    <button class="coder-create-btn" id="coder-retry-btn">Retry</button>
  `;
  currentCoder().dom.terminalPane.appendChild(el);
  el.querySelector('#coder-retry-btn')?.addEventListener('click', () => {
    currentCoder().terminalId = null;
    currentCoder().chatHistory = [];
    currentCoder().conversation?.clear();
    currentCoder().missionPanel?.clear();
    currentCoder().workspaceId = null;
    _onEnterCoderMode();
  });
}

function _onLeaveCoderMode() {
  _removeCoderTheme();
  _stopWorkspaceStatusPolling();
  _stopServicesPolling();
  _resetPreviewState();
  _resetEditorDiagnostics();
  // Do not call _setWorkbenchTab('terminal') while outside coder mode: that
  // helper intentionally unhides #coder-terminal-pane, which makes the pane a
  // visible flex child in normal chat/story/analyze layouts.
  _resetWorkbenchTabForHiddenCoder();
  // MCP + power tiles were polling every 30s; without clearing here they
  // keep firing against /v1/mcp/servers and /api/powers/active forever,
  // which piles up behind inference work and surfaces as 502s elsewhere.
  if (window._coderMcpTileTimer) {
    clearInterval(window._coderMcpTileTimer);
    window._coderMcpTileTimer = null;
  }
  if (window._coderPowerTileTimer) {
    clearInterval(window._coderPowerTileTimer);
    window._coderPowerTileTimer = null;
  }
  // Don't destroy — terminals keep running in background
  // Hide coder-specific UI elements
  const extraKeys = document.getElementById('coder-extra-keys');
  if (extraKeys) extraKeys.style.display = 'none';
}

function _startWorkspaceStatusPolling() {
  if (currentCoder().workspaceStatusPoll) return;
  // Refresh workspace select every 10s so status chips (running/exited)
  // follow live Docker state without requiring a page reload.
  currentCoder().workspaceStatusPoll = setInterval(() => {
    // Don't fetch + rebuild DOM while the tab is backgrounded — nothing is
    // visible to update, and it just burns CPU/network/battery (and adds to
    // the jank when the user returns to a pile of queued work).
    if (document.hidden) return;
    // Skip if the user is actively interacting with the dropdown —
    // rebuilding innerHTML would close an open menu mid-click.
    const select = document.getElementById('coder-workspace-select');
    if (select && document.activeElement === select) return;
    _populateWorkspaceSelect();
  }, 10000);
}

function _stopWorkspaceStatusPolling() {
  if (currentCoder().workspaceStatusPoll) {
    clearInterval(currentCoder().workspaceStatusPoll);
    currentCoder().workspaceStatusPoll = null;
  }
}

function _ensureExtraKeys() {
  if (document.getElementById('coder-extra-keys')) return;

  const bar = document.createElement('div');
  bar.className = 'coder-extra-keys';
  bar.id = 'coder-extra-keys';

  const keys = [
    { label: '//', seq: '//', className: 'coder-agent-key' },  // Agent mode prefix
    { label: 'ESC', seq: '\x1b' },
    { label: 'TAB', seq: '\t' },
    { label: 'CTRL', modifier: true },
    { label: '\u2191', seq: '\x1b[A' },
    { label: '\u2193', seq: '\x1b[B' },
    { label: '\u2192', seq: '\x1b[C' },
    { label: '\u2190', seq: '\x1b[D' },
    { label: '|', seq: '|' },
    { label: '/', seq: '/' },
    { label: '~', seq: '~' },
  ];

  let ctrlActive = false;

  for (const key of keys) {
    const btn = document.createElement('button');
    btn.className = 'coder-extra-key' + (key.className ? ` ${key.className}` : '');
    btn.textContent = key.label;
    btn.addEventListener('click', () => {
      if (key.modifier) {
        if (key.label === 'CTRL') {
          ctrlActive = !ctrlActive;
          btn.classList.toggle('active', ctrlActive);
        }
        return;
      }

      let seq = key.seq;
      if (ctrlActive && seq.length === 1) {
        seq = String.fromCharCode(seq.toUpperCase().charCodeAt(0) - 64);
        ctrlActive = false;
        bar.querySelector('.coder-extra-key.active')?.classList.remove('active');
      }

      document.dispatchEvent(new CustomEvent('coder:send-input', {
        detail: { data: seq },
      }));
    });
    bar.appendChild(btn);
  }

  // Insert after terminal pane
  currentCoder().dom.terminalPane?.parentElement?.appendChild(bar);
}

async function _fetchWorkspaces() {
  try {
    const resp = await fetch('/api/coder/workspaces');
    const data = await resp.json();
    return data.workspaces || [];
  } catch {
    return [];
  }
}

function _updateSafeguardsButton() {
  const btn = document.getElementById('coder-safeguards-btn');
  if (!btn) return;
  if (currentCoder().safeguardsEnabled) {
    btn.style.color = '';
    btn.style.opacity = '';
    btn.title = (
      'Safeguards: ON — soft circuit-breakers active (recommended for ' +
      'weaker local models). Click to disable for strong / API-backed ' +
      'models that legitimately run long.'
    );
  } else {
    btn.style.color = 'var(--text-muted)';
    btn.style.opacity = '0.55';
    btn.title = (
      'Safeguards: OFF — soft breakers bypassed. Only the hard ' +
      'iteration ceiling protects against runaway loops. Click to ' +
      're-enable.'
    );
  }
}

async function _refreshActiveSafeguards() {
  if (!currentCoder().workspaceId) {
    currentCoder().safeguardsEnabled = true;
    _updateSafeguardsButton();
    return;
  }
  try {
    const ws = await _fetchWorkspaces();
    const row = ws.find(w => w.id === currentCoder().workspaceId);
    currentCoder().safeguardsEnabled = row ? row.safeguards_enabled !== false : true;
  } catch {
    currentCoder().safeguardsEnabled = true;
  }
  _updateSafeguardsButton();
}

function _updateAlwaysOnButton() {
  const btn = document.getElementById('coder-always-on-btn');
  if (!btn) return;
  if (currentCoder().alwaysOn) {
    btn.style.color = 'var(--accent, #6cf)';
    btn.style.opacity = '';
    btn.title = (
      'Lifecycle: ALWAYS-ON — container stays running across browser ' +
      'sessions (exempt from idle reaper). Use for workspaces hosting ' +
      'a dev server or daemon. Click to switch to ON-DEMAND.'
    );
  } else {
    btn.style.color = 'var(--text-muted)';
    btn.style.opacity = '0.55';
    btn.title = (
      'Lifecycle: ON-DEMAND — container auto-stops after idle timeout ' +
      'when no client is interacting. Files + DB row survive; restart ' +
      'on next chat. Click to switch to ALWAYS-ON.'
    );
  }
}

function _updateLanButton() {
  const btn = document.getElementById('coder-lan-btn');
  if (!btn) return;
  if (currentCoder().lanAccessible) {
    btn.style.color = 'var(--accent, #6cf)';
    btn.style.opacity = '';
    btn.title = (
      'LAN access: ON — this workspace\'s published ports are reachable ' +
      'from any device on your network (0.0.0.0). For external access, ' +
      'use Tailscale, port forwarding, or localtunnel/Cloudflare Tunnel. ' +
      'Click to restrict to localhost only.'
    );
  } else {
    btn.style.color = 'var(--text-muted)';
    btn.style.opacity = '0.55';
    btn.title = (
      'LAN access: OFF — published ports are loopback-only (127.0.0.1). ' +
      'Click to make services reachable from your network.'
    );
  }
}

function _updateLifecycleButtons() {
  // Start and stop are mutually exclusive — only one is meaningful at
  // a time given the current container status. Paused workspaces show
  // Start (the /start route handles unpause too); stopped show Start;
  // running shows Stop.
  const stopBtn = document.getElementById('coder-stop-workspace-btn');
  const startBtn = document.getElementById('coder-start-workspace-btn');
  if (stopBtn) stopBtn.style.display = (currentCoder().status === 'running') ? '' : 'none';
  if (startBtn) {
    const canStart = (currentCoder().status === 'stopped' || currentCoder().status === 'paused');
    startBtn.style.display = canStart ? '' : 'none';
    startBtn.title = (currentCoder().status === 'paused')
      ? 'Resume paused container'
      : 'Start container';
  }
}

// Per-workspace Bug Finder verifier model override.
// Empty string = fall back to the primary model.
let _activeVerifierModel = '';

async function _saveWorkspaceHeavyweightModel(name) {
  // PUT the new value to the workspace, update local state, toast.
  // Reused by the picker (onSelect) and the right-click clear flow.
  if (!currentCoder().workspaceId) {
    showToast('No workspace selected', 'error');
    return;
  }
  try {
    const resp = await fetch(
      `/api/coder/workspaces/${encodeURIComponent(currentCoder().workspaceId)}/bug-finder-verifier`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ verifier_model: (name || '').trim() }),
      },
    );
    if (!resp.ok) { showToast('Save failed', 'error'); return; }
    const data = await resp.json().catch(() => ({}));
    _activeVerifierModel = data.verifier_model || '';
    _updateVerifierButton();
    showToast(
      _activeVerifierModel
        ? `Heavyweight: ${_activeVerifierModel}`
        : 'Heavyweight: cleared',
      'success',
    );
  } catch {
    showToast('Save failed', 'error');
  }
}

function _updateVerifierButton() {
  const btn = document.getElementById('coder-verifier-btn');
  const label = document.getElementById('coder-verifier-label');
  if (!btn || !label) return;
  // Dual-purpose: this slot is the Bug Finder verifier model AND the
  // stagnation-escalation buddy that the main agent hands off to when
  // it gets stuck. Same model selection, two roles.
  if (_activeVerifierModel) {
    label.textContent = 'HVY*';
    btn.style.color = 'var(--accent, #6cf)';
    btn.title = (
      `Heavyweight model: ${_activeVerifierModel}\n\n` +
      'Used for two roles on this workspace:\n' +
      '  • Bug Finder verifier (audit runs)\n' +
      '  • Stagnation escalation — the main agent hands off here ' +
      'when it loops on the same error or makes no progress\n\n' +
      'Click to change. Right-click to clear.'
    );
  } else {
    label.textContent = 'HVY';
    btn.style.color = '';
    btn.title = (
      'Heavyweight model: (not set)\n\n' +
      'Pick a stronger model here to:\n' +
      '  • Override the Bug Finder verifier role\n' +
      '  • Enable auto-escalation when the main agent gets stuck ' +
      '(repeated tool errors, no-progress streaks)\n\n' +
      'Without one set, stuck turns surface the standard error pill ' +
      'and you can retry manually.'
    );
  }
}

/**
 * Load conversation history from server and render in the conversation pane.
 *
 * Race note (2026-04-22): this function is fire-and-forget from
 * _onEnterCoderMode so the UI can render while history loads. That
 * creates two windows where in-flight user input can get clobbered:
 *
 *   1. The user sends a message BEFORE the fetch resolves. Without
 *      guards, the subsequent ``loadHistory`` (which internally calls
 *      ``clear()``) nukes the user's bubble mid-stream.
 *   2. A second workspace-switch fires while the first is still
 *      fetching — can race the same way.
 *
 * Both guarded via the ``getHistory().length > 0`` check at entry and
 * after the await. Symptom before the fix: on refresh, the user's
 * first message appears, then vanishes ~200-500ms later when the
 * fetch comes back — the exact failure mode reported today.
 *
 * The entry-time guard has a second useful property: on page refresh
 * the just-constructed ``currentCoder().conversation`` is always empty, so the
 * pre-fetch ``clear()`` was redundant. The guard now makes that
 * explicit. On workspace SWITCH the guard correctly fails (old
 * workspace's messages are present), the clear runs, and the new
 * workspace's history replaces cleanly.
 */
async function _loadConversation(workspaceId) {
  if (!currentCoder().conversation) return;

  // Entry guard: a non-empty conversation means either (a) we already
  // loaded history on a prior call or (b) the user started typing
  // before this call fired. Either way, don't clobber — bail cleanly.
  // Workspace-switch paths call clear() explicitly before invoking us,
  // so this guard does not interfere with them.
  if (currentCoder().conversation.getHistory().length > 0) return;

  try {
    const resp = await fetch(`/api/coder/conversation/${encodeURIComponent(workspaceId)}`);

    // A load failure (503) is NOT an empty conversation. Leave history
    // untouched and tell the user it's a load problem they can retry,
    // rather than silently presenting a blank slate over real history.
    if (resp.status === 503) {
      showToast('Couldn’t load this conversation — try reopening the workspace', 'error');
      return;
    }

    const data = await resp.json();

    // Post-fetch guard: the user may have typed a message during the
    // fetch window. Re-check before overwriting; if messages exist
    // now, they are the user's in-flight turn — do NOT clobber them
    // with older server-side history, even if that means we discard
    // the history for this visit. The user always gets another refresh
    // if they want to see older turns.
    if (currentCoder().conversation.getHistory().length > 0) return;

    if (data.messages && data.messages.length > 0) {
      currentCoder().conversation.loadHistory(data.messages);
      currentCoder().chatHistory = currentCoder().conversation.getMessagesForLLM();
    }
    _refreshRewindAffordance();
  } catch {
    // Network-level failure — conversation starts empty this visit.
  }
}

/**
 * Save current conversation history to server.
 */
let _saveConvRetries = 0;
const _SAVE_CONV_MAX_RETRIES = 5;
async function _saveConversation(workspaceId) {
  if (!currentCoder().conversation || !workspaceId) return;
  // Flush any pending debounced save for this workspace — we're persisting
  // the current state right now, so a trailing timer would only re-send it.
  if (_saveConvTimer) { clearTimeout(_saveConvTimer); _saveConvTimer = null; }
  try {
    const res = await fetch(`/api/coder/conversation/${encodeURIComponent(workspaceId)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: currentCoder().conversation.getHistory() }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    _saveConvRetries = 0;
  } catch {
    // The save is the only thing carrying freshly-landed tool results back to
    // the server's conversation blob. Dropping it silently is exactly what
    // left restarts/handoffs showing successful tools as "failed". The
    // durable event ledger still heals the blob on the next read, but retry
    // anyway so the store converges promptly (e.g. the server is mid-restart
    // now and back in a few seconds). Backoff, capped — not an infinite loop.
    if (_saveConvRetries < _SAVE_CONV_MAX_RETRIES) {
      _saveConvRetries += 1;
      const delay = Math.min(2000 * _saveConvRetries, 10000);
      if (_saveConvTimer) clearTimeout(_saveConvTimer);
      _saveConvTimer = setTimeout(() => {
        _saveConvTimer = null;
        void _saveConversation(workspaceId);
      }, delay);
    }
  }
}

// Debounced incremental save. Historically the conversation was persisted
// ONLY at onComplete, so a turn interrupted mid-flight (network drop, the
// user leaving and returning, a dead run) reverted on reload to the last
// fully-completed turn — the user's prompt + every tool call after the last
// checkpoint vanished from the record, so they couldn't ask the agent to
// continue (the code changes survived on disk, but the conversation context
// was gone). Saving incrementally — the prompt on send, then after each tool
// result — means an interrupted turn survives; loadHistory already renders a
// tool with a null result as "interrupted" so the thread is continuable.
// Debounced so a tool-call flurry coalesces into one whole-history write
// (~1.2s) instead of stringifying the full history on every result.
let _saveConvTimer = null;
function _saveConversationSoon(workspaceId) {
  if (!currentCoder().conversation || !workspaceId) return;
  if (_saveConvTimer) clearTimeout(_saveConvTimer);
  _saveConvTimer = setTimeout(() => {
    _saveConvTimer = null;
    void _saveConversation(workspaceId);
  }, 1200);
}

/**
 * Poll until the workspace setup script finishes (ready marker exists).
 * Shows setup progress in the terminal while waiting.
 */
async function _waitForReady(workspaceId, maxWaitMs = 120_000) {
  const start = Date.now();
  const tid = currentCoder().terminalId;
  if (tid) Terminal.write(tid, '\x1b[90mSetting up workspace...\x1b[0m\r\n');
  while (Date.now() - start < maxWaitMs) {
    try {
      const resp = await fetch(`/api/coder/workspaces/${workspaceId}/ready`);
      if (resp.ok) {
        const data = await resp.json();
        if (data.ready) {
          // Surface clone outcome. "none" = no git_url was given.
          // "ok" = clone succeeded. "failed" = clone failed (auth,
          // branch missing, network, etc.) and clone_log holds the
          // tail of git's stderr so the user sees what went wrong.
          if (data.clone_status === 'failed') {
            if (tid) {
              Terminal.write(tid, '\r\n\x1b[31mGit clone failed:\x1b[0m\r\n');
              if (data.clone_log) {
                Terminal.write(tid, '\x1b[90m' + String(data.clone_log).replace(/\n/g, '\r\n') + '\x1b[0m\r\n');
              }
              Terminal.write(tid, '\x1b[33mWorkspace created empty. Add a token in Git Settings and use Pull to retry.\x1b[0m\r\n');
            }
            showToast('Clone failed — see terminal for details', 'error', 6000);
          } else if (data.clone_status === 'ok') {
            if (tid) Terminal.write(tid, '\x1b[32mWorkspace ready (repository cloned).\x1b[0m\r\n');
          } else {
            if (tid) Terminal.write(tid, '\x1b[32mWorkspace ready.\x1b[0m\r\n');
          }
          return data;
        }
      }
    } catch { /* server not ready yet */ }
    await new Promise(r => setTimeout(r, 2000));
  }
  if (tid) Terminal.write(tid, '\x1b[33mSetup is taking longer than expected — workspace may still be initializing.\x1b[0m\r\n');
  return null;
}

/**
 * Create a new workspace and connect terminal to it.
 */
export async function createWorkspace(name = 'workspace', options = {}) {
  try {
    const resp = await fetch('/api/coder/workspaces', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, ...options }),
    });
    const data = await resp.json();
    if (data.id) {
      currentCoder().chatHistory = [];
      currentCoder().conversation?.clear();
      currentCoder().missionPanel?.clear();
      currentCoder().workspaceId = data.id;
      _persistActiveWorkspaceId(data.id);
      currentCoder().conversation?.setWorkspaceId(data.id);
      if (currentCoder().dom.filesTitle) currentCoder().dom.filesTitle.textContent = name;
      // Remove empty state if present
      currentCoder().dom.terminalPane?.querySelector('.coder-empty-state')?.remove();
      if (currentCoder().terminalId) Terminal.destroy(currentCoder().terminalId);
      await Terminal.load();
      currentCoder().terminalId = await Terminal.create(currentCoder().dom.terminalPane, data.id);
      Terminal.focus(currentCoder().terminalId);

      // Wait for workspace setup to finish before loading file tree / git status
      // The container writes .augmentum/ready when all setup (including git clone) is done
      const profile = options.tooling_profile || data.tooling_profile || 'browser';
      const setupWaitMs = profile === 'browser'
        ? 600_000
        : profile === 'power'
          ? 300_000
          : 120_000;
      await _waitForReady(data.id, setupWaitMs);
      _populateFileTree(data.id);
      _startGitPolling();
      // Tell the inspector + other surfaces that the active workspace
      // changed so they re-bind to the new container instead of
      // continuing to render data from whatever was active before.
      document.dispatchEvent(new CustomEvent('coder-workspace-changed', {
        detail: { workspaceId: data.id },
      }));
    }
    return data;
  } catch (err) {
    console.error('Failed to create workspace:', err);
    return null;
  }
}

// ---------------------------------------------------------------------------
// File Tree
// ---------------------------------------------------------------------------

// Per-workspace cache of git status badges, keyed by absolute path.
// Refreshed at the start of every root-level _populateFileTree so the
// tree visualization tracks reality after agent edits, manual edits,
// and turn-end checkpoints. Subtree calls reuse the cache so the
// badges land consistently across nested re-expansions.
let _gitStatusByPath = new Map();
let _gitStatusWorkspaceId = '';

async function _refreshGitStatus(workspaceId) {
  if (!workspaceId) {
    _gitStatusByPath = new Map();
    _gitStatusWorkspaceId = '';
    return;
  }
  try {
    const resp = await fetch(
      `/api/coder/workspaces/${encodeURIComponent(workspaceId)}/git/file-status`,
      { credentials: 'include' },
    );
    if (!resp.ok) {
      _gitStatusByPath = new Map();
      _gitStatusWorkspaceId = workspaceId;
      return;
    }
    const data = await resp.json();
    const next = new Map();
    for (const row of data.files || []) {
      if (row && row.path && row.status) next.set(row.path, row.status);
    }
    _gitStatusByPath = next;
    _gitStatusWorkspaceId = workspaceId;
  } catch {
    // Network hiccup — leave previous cache in place so the tree
    // doesn't lose decorations between successful fetches.
  }
}

// Aggregate a directory's status from its descendants. Used to decorate
// folder rows so the user can spot modified subtrees without expanding
// every level. Priority order picks the most attention-grabbing flag:
// conflicts beat untracked beat modified beat added/deleted/renamed.
const _STATUS_PRIORITY = { C: 5, U: 4, M: 3, D: 2, R: 1, A: 1 };
function _folderStatusFor(folderPath) {
  if (!_gitStatusByPath.size) return '';
  const prefix = folderPath.endsWith('/') ? folderPath : folderPath + '/';
  let best = '';
  let bestScore = 0;
  for (const [path, status] of _gitStatusByPath) {
    if (!path.startsWith(prefix)) continue;
    const score = _STATUS_PRIORITY[status] || 0;
    if (score > bestScore) {
      best = status;
      bestScore = score;
    }
  }
  return best;
}

// Generation token for the root file tree. Bumped on every root call so
// concurrent populates (e.g. _onEnterCoderMode + workspace-switch + a
// post-pull refresh, all racing to clear+await+append) don't double-up
// rows. Stale calls bail before touching the DOM. Subtree calls (when
// `container` is passed) don't participate — they append into their own
// expand-on-demand child container, not the root.
let _fileTreeGen = 0;

// Loading skeleton for the file tree — shown the instant a root populate
// starts so the pane never goes blank while the file_list + git-status
// round-trips are in flight. Replaced by real rows when data arrives.
function _renderFileTreeSkeleton(target) {
  const widths = [72, 54, 84, 63, 77, 48, 69, 58];
  target.innerHTML =
    '<div class="coder-file-skeleton">' +
    widths.map(w =>
      `<div class="coder-file-skeleton-row skeleton" style="width:${w}%"></div>`
    ).join('') +
    '</div>';
}

// ── Index-build progress strip ────────────────────────────────────────
// Polls /api/coder/index/{id}/progress while a build runs and renders a
// quiet determinate bar in the Files panel — the one piece of open-time
// idle we can't trim. Self-terminates on done/idle/staleness so it never
// spins forever.
let _indexProgressTimer = null;

function _indexStripEls() {
  return {
    strip: document.getElementById('coder-index-strip'),
    label: document.getElementById('coder-index-strip-label'),
    count: document.getElementById('coder-index-strip-count'),
    fill: document.getElementById('coder-index-strip-fill'),
  };
}

function _renderIndexProgress(p) {
  const { strip, label, count, fill } = _indexStripEls();
  if (!strip) return;
  const total = p.total || 0;
  const done = Math.min(p.done || 0, total);
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  strip.classList.remove('hidden');
  if (p.state === 'done') {
    strip.classList.add('is-done');
    if (label) label.textContent = 'Code index ready';
    if (count) count.textContent = total ? `${total} files` : '';
    if (fill) fill.style.width = '100%';
  } else {
    strip.classList.remove('is-done');
    if (label) label.textContent = 'Indexing codebase…';
    if (count) count.textContent = total ? `${done}/${total}` : '';
    if (fill) fill.style.width = `${pct}%`;
  }
}

function _hideIndexProgress() {
  const { strip } = _indexStripEls();
  if (strip) { strip.classList.add('hidden'); strip.classList.remove('is-done'); }
}

function _stopIndexProgressPoll(hideAfterMs = 0) {
  if (_indexProgressTimer) { clearInterval(_indexProgressTimer); _indexProgressTimer = null; }
  if (hideAfterMs > 0) setTimeout(_hideIndexProgress, hideAfterMs);
  else _hideIndexProgress();
}

function _startIndexProgressPoll(workspaceId) {
  if (!workspaceId) return;
  _stopIndexProgressPoll(); // reset any prior poll + strip state
  let lastDone = -1;
  let staleTicks = 0;
  const tick = async () => {
    // Bail if the user switched away mid-poll — the strip belongs to
    // whatever workspace is now active.
    if (workspaceId !== currentCoder().workspaceId) { _stopIndexProgressPoll(); return; }
    let p;
    try {
      const r = await fetch(`/api/coder/index/${encodeURIComponent(workspaceId)}/progress`);
      if (!r.ok) { _stopIndexProgressPoll(); return; }
      p = await r.json();
    } catch { return; } // transient — retry next tick
    if (!p || p.state === 'idle') { _stopIndexProgressPoll(); return; }
    _renderIndexProgress(p);
    if (p.state === 'done') { _stopIndexProgressPoll(2500); return; }
    // Running — guard against a wedged build that never flips to done.
    if (p.done === lastDone) {
      if (++staleTicks > 30) _stopIndexProgressPoll(2500);
    } else { staleTicks = 0; lastDone = p.done; }
  };
  tick();
  // 1s cadence — skip the fetch + render while backgrounded.
  _indexProgressTimer = setInterval(() => { if (!document.hidden) tick(); }, 1000);
}

// Internal drag payload type for tree drag-to-move. Distinct from the
// browser's native 'Files' type used by external upload drags, so the
// two never collide: the upload handlers gate on 'Files', the move
// handlers gate on this. getData() only works on drop (protected-mode
// rules); during dragover we can only test dataTransfer.types.
const _DRAG_TYPE = 'application/x-coder-path';
function _readInternalDrag(dt) {
  try {
    const raw = dt.getData(_DRAG_TYPE);
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}

// ---------------------------------------------------------------------------
// Multi-select in the file tree (Ctrl/Cmd-click toggle, Shift-click range).
// A selection surfaces a toolbar with bulk move / download / delete. Plain
// click clears the selection and does its normal thing (open / expand), so
// single-click-to-open is untouched.
// ---------------------------------------------------------------------------
const _selectedPaths = new Set();
let _selectAnchorPath = null;

function _visibleFileRows() {
  return Array.from(currentCoder().dom.fileTree?.querySelectorAll('.coder-file-entry') || []);
}
function _setRowSelected(row, on) {
  if (!row) return;
  row.classList.toggle('selected', on);
  if (on) _selectedPaths.add(row.dataset.path);
  else _selectedPaths.delete(row.dataset.path);
}
function _clearSelection() {
  _selectedPaths.clear();
  _selectAnchorPath = null;
  currentCoder().dom.fileTree?.querySelectorAll('.coder-file-entry.selected')
    .forEach((r) => r.classList.remove('selected'));
  _updateSelectionToolbar();
}

// Returns true when the click was a selection gesture (caller then skips
// its default open/expand action).
function _handleSelectionGesture(e, row) {
  if (e.shiftKey && _selectAnchorPath) {
    e.preventDefault();
    const rows = _visibleFileRows();
    const ai = rows.findIndex((r) => r.dataset.path === _selectAnchorPath);
    const ti = rows.indexOf(row);
    if (ai >= 0 && ti >= 0) {
      _selectedPaths.clear();
      rows.forEach((r) => r.classList.remove('selected'));
      const [lo, hi] = ai < ti ? [ai, ti] : [ti, ai];
      for (let i = lo; i <= hi; i++) _setRowSelected(rows[i], true);
    }
    _updateSelectionToolbar();
    return true;
  }
  if (e.ctrlKey || e.metaKey) {
    e.preventDefault();
    _setRowSelected(row, !_selectedPaths.has(row.dataset.path));
    _selectAnchorPath = row.dataset.path;
    _updateSelectionToolbar();
    return true;
  }
  return false;
}

// Snapshot selection as [{path, isDir, name}] BEFORE any mutation — the
// rows re-render on refresh and lose their dataset.
function _selectedItems() {
  return [..._selectedPaths].map((p) => {
    const row = currentCoder().dom.fileTree?.querySelector(
      `.coder-file-entry[data-path="${CSS.escape(p)}"]`);
    return { path: p, isDir: !!row?.classList.contains('is-dir'), name: p.split('/').pop() };
  });
}

function _updateSelectionToolbar() {
  let bar = document.getElementById('coder-selection-bar');
  const count = _selectedPaths.size;
  if (count === 0) { bar?.remove(); return; }
  if (!bar) {
    bar = document.createElement('div');
    bar.id = 'coder-selection-bar';
    bar.className = 'coder-selection-bar';
    const tree = document.getElementById('coder-file-tree');
    tree?.parentElement?.insertBefore(bar, tree);
    bar.addEventListener('click', (e) => {
      const act = e.target.closest('[data-act]')?.dataset.act;
      if (act === 'move') _bulkMove();
      else if (act === 'download') _bulkDownload();
      else if (act === 'delete') _bulkDelete();
      else if (act === 'clear') _clearSelection();
    });
  }
  const label = bar.querySelector('.coder-selection-count');
  if (label) { label.textContent = `${count} selected`; return; }
  bar.innerHTML =
    `<span class="coder-selection-count">${count} selected</span>` +
    `<div class="coder-selection-actions">` +
    `<button data-act="move" title="Move selected to a folder">Move</button>` +
    `<button data-act="download" title="Download selected files">Download</button>` +
    `<button data-act="delete" class="danger" title="Delete selected">Delete</button>` +
    `<button data-act="clear" title="Clear selection" aria-label="Clear selection">✕</button>` +
    `</div>`;
}

async function _bulkDelete() {
  const items = _selectedItems();
  if (!items.length) return;
  const dirs = items.filter((i) => i.isDir).length;
  const msg = dirs
    ? `Move ${items.length} item(s) (incl. ${dirs} folder(s)) to trash?`
    : `Delete ${items.length} file(s)?`;
  if (!confirm(msg)) return;
  const wid = currentCoder().workspaceId;
  const trashIds = [];
  let failed = 0;
  for (const it of items) {
    try {
      const resp = await fetch(`/api/coder/files/${encodeURIComponent(wid)}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: it.path, recursive: it.isDir }),
      });
      const data = await resp.json().catch(() => ({}));
      if (resp.ok && data.trash_id) trashIds.push(data.trash_id);
      else if (!resp.ok) failed++;
      // Close open editor tabs for the deleted path/subtree.
      const prefix = it.path.endsWith('/') ? it.path : it.path + '/';
      for (const f of [...currentCoder().editorFiles]) {
        if (f.path === it.path || f.path.startsWith(prefix)) _closeEditorFile(f.path);
      }
    } catch { failed++; }
  }
  _clearSelection();
  await _populateFileTree(wid);
  const done = items.length - failed;
  if (trashIds.length) {
    showToast(`Deleted ${done} item(s)${failed ? ` · ${failed} failed` : ''}`, failed ? 'warning' : 'success', 6000, {
      action: {
        label: 'Undo all',
        onClick: async () => {
          for (const tid of trashIds) await _restoreTrash(wid, tid);
        },
      },
    });
  } else {
    showToast(`Deleted ${done} item(s)${failed ? ` · ${failed} failed` : ''}`, failed ? 'warning' : 'success');
  }
}

async function _bulkMove() {
  const items = _selectedItems();
  if (!items.length) return;
  const dest = prompt(
    `Move ${items.length} item(s) to which folder?\n(relative to the workspace root — blank for root)`, '');
  if (dest == null) return;
  const clean = dest.trim().replace(/^\/+|\/+$/g, '');
  await _moveMany(currentCoder().workspaceId, items, clean ? `/workspace/${clean}` : '/workspace');
}

// Move many items into destDir. Shared by the bulk-move toolbar and
// multi-drag. Bulk skips conflicts (no per-file prompt) and reports a
// summary, so one collision doesn't block the rest.
async function _moveMany(workspaceId, items, destDir) {
  const clean = destDir.replace(/\/+$/, '') || '/workspace';
  let moved = 0, skipped = 0, failed = 0;
  for (const it of items) {
    const base = it.path.replace(/\/+$/, '').split('/').pop();
    const newPath = `${clean}/${base}`;
    if (newPath === it.path) { skipped++; continue; }
    if (it.isDir && newPath.startsWith(it.path.replace(/\/$/, '') + '/')) { skipped++; continue; }
    try {
      const resp = await fetch(`/api/coder/files/${encodeURIComponent(workspaceId)}/rename`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ old_path: it.path, new_path: newPath }),
      });
      if (resp.status === 409) { skipped++; continue; } // exists — skip, no prompt
      if (!resp.ok) { failed++; continue; }
      _retargetEditorPaths(it.path, newPath, it.isDir);
      moved++;
    } catch { failed++; }
  }
  _clearSelection();
  await _populateFileTree(workspaceId);
  const parts = [`Moved ${moved}`];
  if (skipped) parts.push(`${skipped} skipped (already exist)`);
  if (failed) parts.push(`${failed} failed`);
  showToast(parts.join(' · '), failed ? 'warning' : 'success');
}

function _bulkDownload() {
  const items = _selectedItems();
  const files = items.filter((i) => !i.isDir);
  const dirs = items.length - files.length;
  if (!files.length) {
    showToast('Select files to download (folders: use workspace Export)', 'warning');
    return;
  }
  // Stagger the anchor clicks — browsers throttle rapid programmatic
  // downloads and may drop simultaneous ones.
  files.forEach((f, i) => {
    setTimeout(() => _downloadFile(currentCoder().workspaceId, f.path, f.name), i * 250);
  });
  showToast(
    `Downloading ${files.length} file(s)${dirs ? ` · ${dirs} folder(s) skipped` : ''}`,
    'info',
  );
}

async function _populateFileTree(workspaceId, path = '/workspace', container = null, depth = 0) {
  const target = container || currentCoder().dom.fileTree;
  if (!target) return;

  let myGen = 0;
  let gitStatusPromise = null;
  if (!container) {
    // Root call — clear existing content, bump generation, and kick off the
    // git-status refresh CONCURRENTLY with the file_list fetch below. It used
    // to be awaited here first, serializing two independent round-trips (on a
    // large working tree `git status --porcelain` adds its full latency before
    // the file list even starts). We hold the promise and await it just before
    // rendering so the first paint still carries decorations.
    myGen = ++_fileTreeGen;
    _renderFileTreeSkeleton(target);
    gitStatusPromise = _refreshGitStatus(workspaceId);
  }

  try {
    let resp = await fetch(`/api/coder/files/${encodeURIComponent(workspaceId)}?path=${encodeURIComponent(path)}`);
    if (!container && myGen !== _fileTreeGen) return;
    // If /workspace doesn't exist yet, fall back to /root
    if (!resp.ok && path === '/workspace' && !container) {
      resp = await fetch(`/api/coder/files/${encodeURIComponent(workspaceId)}?path=${encodeURIComponent('/root')}`);
      if (myGen !== _fileTreeGen) return;
    }
    if (!resp.ok) {
      if (!container && myGen === _fileTreeGen) {
        const errData = await resp.json().catch(() => ({}));
        target.innerHTML = `<p class="text-muted" style="padding:var(--space-sm);font-size:var(--text-xs)">${escapeHtml(errData.error || 'Container not running. Start the workspace first.')}</p>`;
      }
      return;
    }
    const data = await resp.json();
    if (!container && myGen !== _fileTreeGen) return;
    // Block on the git-status cache only now — it raced the file_list fetch
    // above, so by here it has usually already resolved (and never rejects;
    // it swallows its own errors). Re-check generation after the await.
    if (gitStatusPromise) {
      await gitStatusPromise;
      if (myGen !== _fileTreeGen) return;
    }
    const files = data.files || [];

    // Data + git decorations are ready — swap the loading skeleton for the
    // real rows. (Root call only; subtree calls render into their own
    // child container which never held a skeleton.)
    if (!container) target.innerHTML = '';

    // Sort: directories first, then alphabetical
    files.sort((a, b) => {
      if (a.is_dir !== b.is_dir) return a.is_dir ? -1 : 1;
      return a.name.localeCompare(b.name);
    });

    for (const entry of files) {
      // Skip hidden files at root level
      if (depth === 0 && entry.name.startsWith('.') && entry.name !== '.gitignore') continue;

      const row = document.createElement('div');
      row.className = 'coder-file-entry' + (entry.is_dir ? ' is-dir' : '');
      row.style.setProperty('--depth', depth);
      row.dataset.path = entry.path;

      const icon = entry.is_dir ? '\u25B8' : _fileIcon(entry.name);
      const gitStatus = entry.is_dir
        ? _folderStatusFor(entry.path)
        : (_gitStatusByPath.get(entry.path) || '');
      const badgeHtml = gitStatus
        ? `<span class="coder-file-git-badge" data-status="${gitStatus}" title="${_gitStatusLabel(gitStatus)}">${gitStatus}</span>`
        : '';
      const sizeHtml = (!entry.is_dir && entry.size)
        ? `<span class="coder-file-size">${_formatFileSize(entry.size)}</span>`
        : '';
      row.innerHTML =
        `<span class="coder-file-icon">${icon}</span>` +
        `<span class="coder-file-name">${escapeHtml(entry.name)}</span>` +
        badgeHtml + sizeHtml;
      if (gitStatus) row.dataset.gitStatus = gitStatus;

      // Every row is a drag SOURCE for move. The payload carries the
      // path + kind; dragstart stops propagation so a nested row's drag
      // doesn't also start its ancestor's.
      row.draggable = true;
      row.addEventListener('dragstart', (ev) => {
        ev.stopPropagation();
        ev.dataTransfer.effectAllowed = 'move';
        // Dragging a row that's part of a multi-selection moves the whole
        // selection; otherwise just this one.
        const multi = _selectedPaths.has(entry.path) && _selectedPaths.size > 1;
        const payload = multi
          ? { multi: _selectedItems() }
          : { path: entry.path, isDir: entry.is_dir, name: entry.name };
        ev.dataTransfer.setData(_DRAG_TYPE, JSON.stringify(payload));
        row.classList.add('coder-file-entry--dragging');
      });
      row.addEventListener('dragend', () => {
        row.classList.remove('coder-file-entry--dragging');
      });

      if (entry.is_dir) {
        let expanded = false;
        const childContainer = document.createElement('div');
        childContainer.className = 'coder-file-children';
        childContainer.style.display = 'none';

        row.addEventListener('click', async (e) => {
          e.stopPropagation();
          if (_handleSelectionGesture(e, row)) return;
          if (_selectedPaths.size) _clearSelection();
          expanded = !expanded;
          row.classList.toggle('expanded', expanded);
          childContainer.style.display = expanded ? '' : 'none';
          if (expanded && childContainer.children.length === 0) {
            await _populateFileTree(workspaceId, entry.path, childContainer, depth + 1);
          }
        });
        // Right-click → folder context menu (new file/folder, rename,
        // delete, copy path). Mirrors file context menu; stopPropagation
        // keeps a click on the folder row from also triggering the
        // expand/collapse behavior bound above.
        row.addEventListener('contextmenu', (e) => {
          e.preventDefault();
          e.stopPropagation();
          _showFolderContextMenu(e, workspaceId, entry.path, entry.name);
        });

        // Folders are internal-move DROP targets. Gate strictly on our
        // custom type so external upload drags (type 'Files') fall
        // through to the tree-level upload handler untouched.
        row.addEventListener('dragover', (ev) => {
          if (!ev.dataTransfer.types.includes(_DRAG_TYPE)) return;
          ev.preventDefault();
          ev.stopPropagation();
          ev.dataTransfer.dropEffect = 'move';
          row.classList.add('coder-file-entry--drop-target');
        });
        row.addEventListener('dragleave', () => {
          row.classList.remove('coder-file-entry--drop-target');
        });
        row.addEventListener('drop', async (ev) => {
          if (!ev.dataTransfer.types.includes(_DRAG_TYPE)) return;
          ev.preventDefault();
          ev.stopPropagation();
          row.classList.remove('coder-file-entry--drop-target');
          const payload = _readInternalDrag(ev.dataTransfer);
          if (payload?.multi) {
            await _moveMany(workspaceId, payload.multi, entry.path);
          } else if (payload) {
            await _moveEntry(workspaceId, payload.path, entry.path, { isDir: payload.isDir });
          }
        });

        target.appendChild(row);
        target.appendChild(childContainer);
      } else {
        row.addEventListener('click', (e) => {
          e.stopPropagation();
          if (_handleSelectionGesture(e, row)) return;
          if (_selectedPaths.size) _clearSelection();
          // Highlight active file
          currentCoder().dom.fileTree.querySelectorAll('.coder-file-entry.active').forEach(el => el.classList.remove('active'));
          row.classList.add('active');
          _openFileInEditor(currentCoder().workspaceId, entry.path, entry.name);
        });
        row.addEventListener('contextmenu', (e) => {
          e.preventDefault();
          _showFileContextMenu(e, workspaceId, entry.path, entry.name);
        });
        target.appendChild(row);
      }
    }
  } catch (err) {
    console.error('File tree fetch failed:', err);
  }
}

function _fileIcon(name) {
  // Simple monospace-friendly icons — no emoji
  const ext = name.split('.').pop()?.toLowerCase();
  const icons = {
    js: 'JS', ts: 'TS', py: 'PY', md: 'MD', json: '{}',
    html: '<>', css: '#', sh: '$', yml: '~', yaml: '~',
    txt: 'T', rs: 'RS', go: 'GO', c: 'C', cpp: 'C+', h: '.h',
    java: 'JV', rb: 'RB', toml: '~', xml: '<>',
    lock: 'LK', gitignore: '.g',
  };
  return icons[ext] || '\u{25CB}';
}

// Human-readable label for a git status code. Surfaced on the badge's
// hover title so vibe coders learn what M/U/D mean without having to
// click into the git panel — keeps the badge itself a single letter
// so it doesn't dominate the row.
function _gitStatusLabel(status) {
  switch (status) {
    case 'M': return 'Modified — uncommitted changes';
    case 'U': return 'Untracked — new file, not in git yet';
    case 'A': return 'Added — staged for commit';
    case 'D': return 'Deleted — removal pending';
    case 'R': return 'Renamed — staged with new path';
    case 'C': return 'Conflict — needs manual resolution';
    default:  return '';
  }
}

function _getCodeMindLanguage(filePath = '') {
  const ext = filePath.split('.').pop()?.toLowerCase() || '';
  const langMap = {
    js: 'javascript',
    jsx: 'javascript',
    ts: 'typescript',
    tsx: 'typescript',
    py: 'python',
    html: 'html',
    htm: 'html',
    css: 'css',
    scss: 'css',
    json: 'json',
  };
  return langMap[ext] || null;
}

function _getEditorDiagnosticsEl() {
  return currentCoder().dom.editorPane?.querySelector('.coder-editor-diagnostics') || null;
}

function _updateEditorDiagnosticsStatus(state = 'hidden', errors = []) {
  const badge = _getEditorDiagnosticsEl();
  if (!badge) return;

  badge.classList.remove('hidden', 'checking', 'ok', 'error');

  if (state === 'hidden') {
    badge.classList.add('hidden');
    badge.textContent = '';
    badge.title = '';
    badge.disabled = true;
    return;
  }

  badge.disabled = state !== 'error' || errors.length === 0;

  if (state === 'checking') {
    badge.classList.add('checking');
    badge.textContent = 'Checking…';
    badge.title = 'Loading syntax diagnostics…';
    return;
  }

  if (state === 'ok') {
    badge.classList.add('ok');
    badge.textContent = 'Syntax OK';
    badge.title = 'No syntax errors detected';
    return;
  }

  badge.classList.add('error');
  const count = errors.length;
  badge.textContent = `${count} syntax issue${count === 1 ? '' : 's'}`;
  badge.title = errors.map((err) =>
    `Ln ${Number(err.startRow || 0) + 1}: ${err.message || 'Syntax error'}`
  ).join('\n');
}

async function _ensureCodeMindReady() {
  if (currentCoder().codeMindReady) return true;
  if (currentCoder().codeMindInitPromise) return currentCoder().codeMindInitPromise;

  currentCoder().codeMindInitPromise = (async () => {
    try {
      const ok = await CodeMind.init();
      currentCoder().codeMindReady = !!ok;
      return currentCoder().codeMindReady;
    } catch {
      currentCoder().codeMindReady = false;
      return false;
    } finally {
      if (!currentCoder().codeMindReady) currentCoder().codeMindInitPromise = null;
    }
  })();

  return currentCoder().codeMindInitPromise;
}

function _focusFirstEditorDiagnostic() {
  if (!currentCoder().activeEditorId || currentCoder().activeEditorDiagnostics.length === 0) return;
  const first = currentCoder().activeEditorDiagnostics[0] || {};
  const line = Number(first.startRow || 0) + 1;
  const col = Number(first.startCol || 0) + 1;
  Editor.setCursor(currentCoder().activeEditorId, line, col);
  Editor.focus(currentCoder().activeEditorId);
}

function _resetEditorDiagnostics() {
  if (currentCoder().codeMindDebounce) {
    clearTimeout(currentCoder().codeMindDebounce);
    currentCoder().codeMindDebounce = null;
  }
  currentCoder().activeEditorDiagnosticsToken += 1;
  currentCoder().activeEditorCodeMindLanguage = null;
  currentCoder().activeEditorDiagnostics = [];
  if (currentCoder().activeEditorId) Editor.setDiagnosticsFromCodeMind(currentCoder().activeEditorId, []);
  _updateEditorDiagnosticsStatus('hidden');
}

function _runEditorDiagnostics() {
  if (!currentCoder().codeMindReady || !currentCoder().activeEditorId || !currentCoder().activeEditorCodeMindLanguage || !currentCoder().activeFilePath) return;
  const content = Editor.getContent(currentCoder().activeEditorId);
  const result = CodeMind.parseSync(content, currentCoder().activeEditorCodeMindLanguage, currentCoder().activeFilePath);
  currentCoder().activeEditorDiagnostics = result ? (result.errors || []) : [];
  Editor.setDiagnosticsFromCodeMind(currentCoder().activeEditorId, currentCoder().activeEditorDiagnostics);
  _updateEditorDiagnosticsStatus(currentCoder().activeEditorDiagnostics.length > 0 ? 'error' : 'ok', currentCoder().activeEditorDiagnostics);
}

function _queueEditorDiagnostics() {
  if (!currentCoder().activeEditorCodeMindLanguage) {
    _updateEditorDiagnosticsStatus('hidden');
    return;
  }
  if (!currentCoder().codeMindReady) return;
  if (currentCoder().codeMindDebounce) clearTimeout(currentCoder().codeMindDebounce);
  currentCoder().codeMindDebounce = setTimeout(_runEditorDiagnostics, 150);
}

async function _primeEditorDiagnostics(content, filePath, token) {
  if (!currentCoder().activeEditorCodeMindLanguage) {
    _updateEditorDiagnosticsStatus('hidden');
    return;
  }
  _updateEditorDiagnosticsStatus('checking');
  const ok = await _ensureCodeMindReady();
  if (!ok) {
    _updateEditorDiagnosticsStatus('hidden');
    return;
  }
  if (token !== currentCoder().activeEditorDiagnosticsToken || filePath !== currentCoder().activeEditorFile) {
    return;
  }
  await CodeMind.parse(content || '', currentCoder().activeEditorCodeMindLanguage, filePath).catch(() => null);
  if (token !== currentCoder().activeEditorDiagnosticsToken || filePath !== currentCoder().activeEditorFile) {
    return;
  }
  _runEditorDiagnostics();
}

// ---------------------------------------------------------------------------
// Context Menu
// ---------------------------------------------------------------------------

function _showFileContextMenu(e, workspaceId, filePath, fileName) {
  const items = [
    { label: 'Open', action: () => _openFileInEditor(currentCoder().workspaceId, filePath, fileName) },
  ];
  // Preview is only meaningful for file types we can render in an iframe.
  // isPreviewable returns false until the server registry is fetched;
  // we kick that off at coder-mode init so this is populated by the
  // time the user right-clicks anything.
  if (isPreviewable(fileName)) {
    items.push({
      label: 'Preview',
      action: () => _openFilePreview(workspaceId, filePath, fileName),
    });
  }
  // "View changes" only when the file actually differs from the last
  // commit (has a git badge) — a diff of an unmodified file is noise.
  // Untracked (U) files ARE offered: the modal shows them as a full
  // addition, which is exactly what you want after a model writes a
  // new script.
  if (_gitStatusByPath.get(filePath)) {
    items.push({
      label: 'View changes',
      action: () => _showFileDiff(workspaceId, filePath, fileName),
    });
  }
  items.push(
    { divider: true },
    { label: 'Rename…', action: () => _beginInlineRename(_rowForPath(filePath), workspaceId, filePath, fileName, false) },
    { label: 'Move to…', action: () => _promptMoveEntry(workspaceId, filePath, fileName, false) },
    { label: 'Duplicate', action: () => _duplicateFile(workspaceId, filePath, fileName) },
    { label: 'Download', action: () => _downloadFile(workspaceId, filePath, fileName) },
    { label: 'Copy path', action: () => _copyToClipboard(filePath, 'Path copied') },
    { divider: true },
    { label: 'Delete', danger: true, action: () => _deleteFilePath(workspaceId, filePath, fileName, false) },
  );
  _renderContextMenu(e, items);
}

// Stream a single file down as an attachment via the (previously
// UI-less) download endpoint. Binary-safe on the server side.
function _downloadFile(workspaceId, filePath, fileName) {
  const url = `/api/coder/files/${encodeURIComponent(workspaceId)}/download?path=${encodeURIComponent(filePath)}`;
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName || '';
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function _showFolderContextMenu(e, workspaceId, folderPath, folderName) {
  const items = [
    { label: 'New file…', action: () => _beginInlineCreate(workspaceId, folderPath, false) },
    { label: 'New folder…', action: () => _beginInlineCreate(workspaceId, folderPath, true) },
    { divider: true },
    { label: 'Rename…', action: () => _beginInlineRename(_rowForPath(folderPath), workspaceId, folderPath, folderName, true) },
    { label: 'Move to…', action: () => _promptMoveEntry(workspaceId, folderPath, folderName, true) },
    { label: 'Copy path', action: () => _copyToClipboard(folderPath, 'Path copied') },
    { divider: true },
    { label: 'Delete folder…', danger: true, action: () => _deleteFilePath(workspaceId, folderPath, folderName, true) },
  ];
  _renderContextMenu(e, items);
}

// Shared menu renderer — keeps the file + folder menus visually
// identical and handles outside-click dismissal once.
function _renderContextMenu(e, items) {
  document.querySelector('.coder-context-menu')?.remove();

  const menu = document.createElement('div');
  menu.className = 'coder-context-menu';
  // Defer the final position until after we have the rendered size so
  // a menu opened near the bottom-right of the viewport doesn't clip.
  menu.style.left = e.clientX + 'px';
  menu.style.top = e.clientY + 'px';
  menu.style.visibility = 'hidden';

  for (const item of items) {
    if (item.divider) {
      const sep = document.createElement('div');
      sep.className = 'coder-context-divider';
      menu.appendChild(sep);
      continue;
    }
    const row = document.createElement('div');
    row.className = 'coder-context-item' + (item.danger ? ' danger' : '');
    row.textContent = item.label;
    row.onclick = () => { menu.remove(); item.action(); };
    menu.appendChild(row);
  }

  document.body.appendChild(menu);
  // Flip into viewport if needed.
  const rect = menu.getBoundingClientRect();
  if (rect.right > window.innerWidth - 8) {
    menu.style.left = Math.max(8, window.innerWidth - rect.width - 8) + 'px';
  }
  if (rect.bottom > window.innerHeight - 8) {
    menu.style.top = Math.max(8, window.innerHeight - rect.height - 8) + 'px';
  }
  menu.style.visibility = 'visible';

  // Close on outside click. Add the listener on the NEXT tick so the
  // contextmenu event that opened the menu doesn't immediately dismiss it.
  setTimeout(() => {
    const handler = (ev) => {
      if (!menu.contains(ev.target)) {
        menu.remove();
        document.removeEventListener('click', handler);
        document.removeEventListener('contextmenu', handler);
      }
    };
    document.addEventListener('click', handler);
    document.addEventListener('contextmenu', handler);
  }, 0);
}

// ---------------------------------------------------------------------------
// File-tree operations (backed by the file-management routes added 2026-05-31)
// ---------------------------------------------------------------------------

// Shared rename/move-with-explicit-target executor: POSTs the rename,
// handles the destination-exists 409 (confirm→overwrite), and retargets
// open tabs so both rename and drag/prompt-move keep editor state.
async function _renameOrMoveTo(workspaceId, oldPath, newPath, label, isDir, overwrite = false) {
  try {
    const resp = await fetch(
      `/api/coder/files/${encodeURIComponent(workspaceId)}/rename`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ old_path: oldPath, new_path: newPath, overwrite }),
      },
    );
    if (resp.status === 409) {
      if (confirm(`"${label}" already exists.\n\nOverwrite it?`)) {
        return _renameOrMoveTo(workspaceId, oldPath, newPath, label, isDir, true);
      }
      return;
    }
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      showToast(extractErrorMessage(err, 'Rename failed'), 'error');
      return;
    }
    // Keep open tabs pointing at the new path (files AND descendants of
    // a renamed folder), preserving unsaved edits.
    _retargetEditorPaths(oldPath, newPath, isDir);
    showToast(`Renamed to ${label}`, 'success');
    await _populateFileTree(workspaceId);
  } catch {
    showToast('Rename failed', 'error');
  }
}

function _rowForPath(path) {
  return currentCoder().dom.fileTree?.querySelector(
    `.coder-file-entry[data-path="${CSS.escape(path)}"]`) || null;
}

function _formatFileSize(n) {
  if (!n || n < 0) return '';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(n < 10240 ? 1 : 0)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

// Reveal & follow the active file: expand its ancestor folders top-down
// (lazy-loading each), then highlight the row and scroll it into view.
// Cheap when the file is already visible (no collapsed ancestors → no
// expansion, just re-highlight + scroll-nearest).
async function _revealInTree(path) {
  if (!currentCoder().dom.fileTree || !path) return;
  const rel = path.replace(/^\/workspace\/?/, '');
  const parts = rel.split('/').filter(Boolean);
  parts.pop(); // ancestors only — drop the file name
  let cur = '/workspace';
  for (const seg of parts) {
    cur = `${cur}/${seg}`;
    const frow = _rowForPath(cur);
    if (!frow) break;
    const child = frow.nextElementSibling;
    if (child && child.classList.contains('coder-file-children') &&
        child.style.display === 'none') {
      frow.click();              // expand + trigger lazy load
      await _waitForChildren(child);
    }
  }
  const row = _rowForPath(path);
  if (row) {
    currentCoder().dom.fileTree.querySelectorAll('.coder-file-entry.active')
      .forEach((el) => el.classList.remove('active'));
    row.classList.add('active');
    row.scrollIntoView({ block: 'nearest' });
  }
}

function _waitForChildren(child, tries = 20) {
  return new Promise((resolve) => {
    let n = 0;
    const tick = () => {
      if (child.children.length > 0 || ++n >= tries) resolve();
      else setTimeout(tick, 25);
    };
    tick();
  });
}

// Inline rename — edit the name in place instead of a browser prompt().
// Enter / blur commits, Esc cancels. Files pre-select the base name (not
// the extension), matching editor muscle memory.
function _beginInlineRename(row, workspaceId, path, name, isDir) {
  if (!row || row.querySelector('.coder-inline-input')) return;
  const nameSpan = row.querySelector('.coder-file-name');
  if (!nameSpan) return;
  const input = document.createElement('input');
  input.className = 'coder-inline-input';
  input.value = name;
  input.spellcheck = false;
  nameSpan.style.display = 'none';
  nameSpan.after(input);

  let done = false;
  const finish = async (commit) => {
    if (done) return;
    done = true;
    const val = input.value.trim();
    input.remove();
    nameSpan.style.display = '';
    if (!commit || !val || val === name) return;
    if (val.includes('/') || val.includes('\\')) {
      showToast('Name cannot contain slashes', 'error');
      return;
    }
    const dir = path.substring(0, path.lastIndexOf('/'));
    await _renameOrMoveTo(workspaceId, path, `${dir}/${val}`, val, isDir);
  };
  input.addEventListener('keydown', (e) => {
    e.stopPropagation();
    if (e.key === 'Enter') { e.preventDefault(); finish(true); }
    else if (e.key === 'Escape') { e.preventDefault(); finish(false); }
  });
  input.addEventListener('blur', () => finish(true));
  input.addEventListener('click', (e) => e.stopPropagation());
  input.addEventListener('dblclick', (e) => e.stopPropagation());
  setTimeout(() => {
    input.focus();
    const dot = name.lastIndexOf('.');
    if (!isDir && dot > 0) input.setSelectionRange(0, dot);
    else input.select();
  }, 0);
}

// Inline create — a placeholder editable row instead of a prompt(). Roots
// at the tree top or inside a folder (expanding it first). Enter/blur
// commits; Esc or an empty value removes the row.
async function _beginInlineCreate(workspaceId, folderPath, isDir) {
  if (!workspaceId) { showToast('No workspace selected', 'warning'); return; }
  if (!currentCoder().dom.fileTree) return;
  const isRoot = !folderPath || folderPath === '/workspace';
  let container = currentCoder().dom.fileTree;
  let depth = 0;
  if (!isRoot) {
    const frow = _rowForPath(folderPath);
    const child = frow?.nextElementSibling;
    if (child && child.classList.contains('coder-file-children')) {
      if (child.style.display === 'none') frow.click(); // expand + lazy-load
      container = child;
      depth = (parseInt(frow.style.getPropertyValue('--depth'), 10) || 0) + 1;
    }
  }

  const row = document.createElement('div');
  row.className = 'coder-file-entry coder-file-entry--editing';
  row.style.setProperty('--depth', depth);
  row.innerHTML = `<span class="coder-file-icon">${isDir ? '▸' : '○'}</span>`;
  const input = document.createElement('input');
  input.className = 'coder-inline-input';
  input.placeholder = isDir ? 'folder name' : 'file name';
  input.spellcheck = false;
  row.appendChild(input);
  container.insertBefore(row, container.firstChild);

  let done = false;
  const finish = async (commit) => {
    if (done) return;
    done = true;
    const val = input.value.trim();
    row.remove();
    if (!commit || !val) return;
    if (val.includes('/') || val.includes('\\')) {
      showToast('Name cannot contain slashes', 'error');
      return;
    }
    const base = (isRoot ? '/workspace' : folderPath).replace(/\/$/, '');
    const newPath = `${base}/${val}`;
    try {
      if (isDir) {
        const r = await fetch(`/api/coder/files/${encodeURIComponent(workspaceId)}/mkdir`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: newPath }),
        });
        if (!r.ok) {
          showToast(extractErrorMessage(await r.json().catch(() => ({})), 'Create failed'), 'error');
          return;
        }
        await _populateFileTree(workspaceId);
      } else {
        const r = await fetch(`/api/coder/files/${encodeURIComponent(workspaceId)}/write`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: newPath, content: '' }),
        });
        if (!r.ok) {
          showToast(extractErrorMessage(await r.json().catch(() => ({})), 'Create failed'), 'error');
          return;
        }
        await _populateFileTree(workspaceId);
        _openFileInEditor(workspaceId, newPath, val);
      }
    } catch {
      showToast('Create failed', 'error');
    }
  };
  input.addEventListener('keydown', (e) => {
    e.stopPropagation();
    if (e.key === 'Enter') { e.preventDefault(); finish(true); }
    else if (e.key === 'Escape') { e.preventDefault(); finish(false); }
  });
  input.addEventListener('blur', () => finish(true));
  input.addEventListener('click', (e) => e.stopPropagation());
  setTimeout(() => input.focus(), 0);
}

async function _duplicateFile(workspaceId, filePath, fileName) {
  const dot = fileName.lastIndexOf('.');
  const base = dot > 0 ? fileName.slice(0, dot) : fileName;
  const ext = dot > 0 ? fileName.slice(dot) : '';
  const copyName = `${base} copy${ext}`;
  const dir = filePath.substring(0, filePath.lastIndexOf('/'));
  const newPath = `${dir}/${copyName}`;
  try {
    const readResp = await fetch(
      `/api/coder/files/${encodeURIComponent(workspaceId)}/read?path=${encodeURIComponent(filePath)}`,
    );
    if (!readResp.ok) { showToast('Failed to read source', 'error'); return; }
    const content = (await readResp.json()).content || '';
    const writeResp = await fetch(
      `/api/coder/files/${encodeURIComponent(workspaceId)}/write`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: newPath, content }),
      },
    );
    if (!writeResp.ok) { showToast('Failed to write copy', 'error'); return; }
    showToast(`Duplicated → ${copyName}`, 'success');
    await _populateFileTree(workspaceId);
  } catch {
    showToast('Duplicate failed', 'error');
  }
}

async function _deleteFilePath(workspaceId, path, name, isDir) {
  // Deletes are now reversible (move to trash + Undo). Folders still
  // confirm — trashing a whole subtree is a bigger action — but the
  // wording reflects recoverability. Single files skip the confirm; the
  // Undo toast is the safety net.
  if (isDir && !confirm(`Move folder "${name}" and its contents to trash?`)) return;
  try {
    const resp = await fetch(
      `/api/coder/files/${encodeURIComponent(workspaceId)}`,
      {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path, recursive: isDir }),
      },
    );
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      showToast(extractErrorMessage(err, 'Delete failed'), 'error');
      return;
    }
    const data = await resp.json().catch(() => ({}));
    // Close any open editor tabs pointing at the deleted path/subtree.
    if (!isDir && currentCoder().editorFiles.find((f) => f.path === path)) {
      _closeEditorFile(path);
    } else if (isDir) {
      const prefix = path.endsWith('/') ? path : path + '/';
      for (const f of [...currentCoder().editorFiles]) {
        if (f.path.startsWith(prefix)) _closeEditorFile(f.path);
      }
    }
    await _populateFileTree(workspaceId);
    if (data.trashed && data.trash_id) {
      showToast(`Deleted ${name}`, 'success', 6000, {
        action: { label: 'Undo', onClick: () => _restoreTrash(workspaceId, data.trash_id) },
      });
    } else {
      showToast(`Deleted ${name}`, 'success');
    }
  } catch {
    showToast('Delete failed', 'error');
  }
}

// Restore a trashed item to its original path (delete Undo / trash drawer).
async function _restoreTrash(workspaceId, trashId) {
  try {
    const resp = await fetch(
      `/api/coder/files/${encodeURIComponent(workspaceId)}/restore`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trash_id: trashId }),
      },
    );
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      // 409 = original path now occupied, or entry gone. Surface why.
      showToast(data.reason || 'Restore failed', 'error');
      return;
    }
    showToast('Restored', 'success');
    await _populateFileTree(workspaceId);
  } catch {
    showToast('Restore failed', 'error');
  }
}

async function _createNewFileIn(workspaceId, folderPath) {
  const name = prompt('New file name:', '');
  if (name == null) return;
  const trimmed = name.trim();
  if (!trimmed) return;
  if (trimmed.includes('/') || trimmed.includes('\\')) {
    showToast('Name cannot contain slashes', 'error');
    return;
  }
  const newPath = `${folderPath.replace(/\/$/, '')}/${trimmed}`;
  try {
    const resp = await fetch(
      `/api/coder/files/${encodeURIComponent(workspaceId)}/write`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: newPath, content: '' }),
      },
    );
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      showToast(extractErrorMessage(err, 'Create failed'), 'error');
      return;
    }
    await _populateFileTree(workspaceId);
    _openFileInEditor(workspaceId, newPath, trimmed);
  } catch {
    showToast('Create failed', 'error');
  }
}

async function _createNewFolderIn(workspaceId, folderPath) {
  const name = prompt('New folder name:', '');
  if (name == null) return;
  const trimmed = name.trim();
  if (!trimmed) return;
  if (trimmed.includes('/') || trimmed.includes('\\')) {
    showToast('Name cannot contain slashes', 'error');
    return;
  }
  const newPath = `${folderPath.replace(/\/$/, '')}/${trimmed}`;
  try {
    const resp = await fetch(
      `/api/coder/files/${encodeURIComponent(workspaceId)}/mkdir`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: newPath }),
      },
    );
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      showToast(extractErrorMessage(err, 'Create failed'), 'error');
      return;
    }
    await _populateFileTree(workspaceId);
  } catch {
    showToast('Create failed', 'error');
  }
}

// Retarget open editor tabs when a path is renamed or moved, in place —
// preserving unsaved buffer content — instead of closing them (the old
// rename behavior). Handles a moved/renamed DIRECTORY by remapping every
// open descendant. Fixes the class: rename AND move both keep your tabs.
function _retargetEditorPaths(oldPath, newPath, isDir) {
  const prefix = oldPath.endsWith('/') ? oldPath : oldPath + '/';
  const newBase = newPath.replace(/\/$/, '');
  const remap = (p) => {
    if (p === oldPath) return newPath;
    if (isDir && p.startsWith(prefix)) return `${newBase}/${p.slice(prefix.length)}`;
    return null;
  };
  let changed = false;
  for (const f of currentCoder().editorFiles) {
    const np = remap(f.path);
    if (np) {
      f.path = np;
      f.name = np.split('/').pop();
      changed = true;
    }
  }
  const na = currentCoder().activeEditorFile && remap(currentCoder().activeEditorFile);
  if (na) currentCoder().activeEditorFile = na;
  const nf = currentCoder().activeFilePath && remap(currentCoder().activeFilePath);
  if (nf) currentCoder().activeFilePath = nf;
  if (changed) _renderEditorTabs();
}

// Move a file/folder INTO destDir (an absolute /workspace path). Reuses
// the rename endpoint (mv). Handles the destination-exists 409 with a
// confirm→overwrite retry, guards against moving a folder into itself,
// and retargets open tabs. Shared by drag-to-move and "Move to…".
async function _moveEntry(workspaceId, srcPath, destDir, { isDir = false, overwrite = false } = {}) {
  const base = srcPath.replace(/\/+$/, '').split('/').pop();
  const cleanDir = destDir.replace(/\/+$/, '') || '/workspace';
  const newPath = `${cleanDir}/${base}`;
  if (newPath === srcPath) return; // dropped onto its own parent — no-op
  if (isDir && (newPath === srcPath || newPath.startsWith(srcPath.replace(/\/$/, '') + '/'))) {
    showToast("Can't move a folder into itself", 'error');
    return;
  }
  try {
    const resp = await fetch(
      `/api/coder/files/${encodeURIComponent(workspaceId)}/rename`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ old_path: srcPath, new_path: newPath, overwrite }),
      },
    );
    if (resp.status === 409) {
      // Destination-exists guard from the backend. Ask before clobbering.
      if (confirm(`"${base}" already exists in ${cleanDir.replace(/^\/workspace\/?/, '') || 'the workspace root'}.\n\nOverwrite it?`)) {
        return _moveEntry(workspaceId, srcPath, destDir, { isDir, overwrite: true });
      }
      return;
    }
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      showToast(extractErrorMessage(err, 'Move failed'), 'error');
      return;
    }
    _retargetEditorPaths(srcPath, newPath, isDir);
    showToast(`Moved ${base}`, 'success');
    await _populateFileTree(workspaceId);
  } catch {
    showToast('Move failed', 'error');
  }
}

// Accessible / mobile fallback for drag-to-move: prompt for a
// destination folder relative to the workspace root.
async function _promptMoveEntry(workspaceId, srcPath, name, isDir) {
  const curDir = srcPath.slice(0, srcPath.lastIndexOf('/'))
    .replace(/^\/workspace\/?/, '');
  const dest = prompt(
    `Move "${name}" to which folder?\n(relative to the workspace root — leave blank for root)`,
    curDir,
  );
  if (dest == null) return;
  const clean = dest.trim().replace(/^\/+|\/+$/g, '');
  const destDir = clean ? `/workspace/${clean}` : '/workspace';
  await _moveEntry(workspaceId, srcPath, destDir, { isDir });
}

function _copyToClipboard(text, successMsg) {
  if (navigator.clipboard?.writeText) {
    navigator.clipboard.writeText(text).then(
      () => showToast(successMsg || 'Copied', 'success'),
      () => showToast('Copy failed', 'error'),
    );
    return;
  }
  // Fallback for older browsers / non-secure contexts.
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    showToast(successMsg || 'Copied', 'success');
  } catch {
    showToast('Copy failed', 'error');
  }
}

// ---------------------------------------------------------------------------
// Markdown Renderer (lightweight, for coder output)
// ---------------------------------------------------------------------------

function _renderCoderMarkdown(text) {
  // Escape HTML first
  let html = escapeHtml(text);
  // Render fenced code blocks
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    return `<pre class="coder-code-block"><code>${code.trim()}</code></pre>`;
  });
  // Render inline code
  html = html.replace(/`([^`]+)`/g, '<code class="coder-inline-code">$1</code>');
  // Render bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Newlines to <br>
  html = html.replace(/\n/g, '<br>');
  return html;
}

// ---------------------------------------------------------------------------
// Editor (slide-in panel)
// ---------------------------------------------------------------------------

function _showEditorSplit() {
  const split = currentCoder().dom.editorSplit;
  if (split) {
    split.classList.remove('hidden');
    // Trigger animation after removing hidden (next frame)
    requestAnimationFrame(() => split.classList.add('visible'));
  }
}

function _hideEditorSplit() {
  const split = currentCoder().dom.editorSplit;
  if (split) {
    split.classList.remove('visible');
    // After animation completes, hide completely
    setTimeout(() => {
      if (!split.classList.contains('visible')) {
        split.classList.add('hidden');
      }
    }, 300);
  }
  // Refit terminal since it regains full width
  if (currentCoder().terminalId) Terminal.fit(currentCoder().terminalId);
}

async function _openFileInEditor(workspaceId, filePath, fileName, jumpTo = null) {
  if (!currentCoder().dom.editorPane) return;

  // Preserve the outgoing editor's buffer + cursor before we tear it down,
  // so switching tabs never silently discards unsaved edits.
  _snapshotActiveEditorBuffer();
  const _existingTab = currentCoder().editorFiles.find((f) => f.path === filePath);
  const _useCachedBuffer = !!(_existingTab && _existingTab.buffer != null);

  try {
    let content;
    let _restoreCursor = null;
    let _startDirty = false;
    if (_useCachedBuffer) {
      // Switching back to an already-open (possibly dirty) file — restore
      // its live buffer, not the on-disk version.
      content = _existingTab.buffer;
      _restoreCursor = _existingTab.cursor || null;
      _startDirty = !!_existingTab.dirty;
    } else {
      const resp = await fetch(`/api/coder/files/${encodeURIComponent(workspaceId)}/read?path=${encodeURIComponent(filePath)}`);
      if (!resp.ok) { showToast('Failed to read file', 'error'); return; }
      content = (await resp.json()).content || '';
    }

    // Detect language from extension
    const ext = fileName.split('.').pop()?.toLowerCase() || '';
    const langMap = {
      js: 'javascript', ts: 'typescript', py: 'python', rs: 'rust', go: 'go',
      html: 'html', htm: 'html', css: 'css', json: 'json', md: 'markdown',
      sh: 'shell', bash: 'shell', yml: 'yaml', yaml: 'yaml',
      c: 'c', cpp: 'cpp', h: 'cpp', java: 'java', rb: 'ruby',
    };
    const language = langMap[ext] || 'text';

    // Companion presence: the open editor file — gives "this file" /
    // "this code" a referent. Best-effort, dedup-windowed.
    import('./architect-observer.js')
      .then(m => m.reportAttention('surface.coder.file_opened', {
        label: fileName || '', path: filePath || '', ref: workspaceId || '',
      }))
      .catch(() => {});
    // "Read this file" handoff — the editor buffer lives in CodeMirror's
    // .cm-content. Lazy getter (read on button press, by which point the
    // editor has loaded). Cleared when the user leaves coder (app.js).
    import('./companion-context.js')
      .then(m => m.setCompanionLoadable('file', fileName || filePath || 'this file', () => ({
        label: fileName || filePath || 'this file',
        content: document.querySelector('.cm-content')?.innerText || '',
        ref: filePath || '',
      })))
      .catch(() => {});

    // Load editor if needed
    await Editor.load();

    _resetEditorDiagnostics();

    // Destroy existing editor
    if (currentCoder().activeEditorId) {
      Editor.destroy(currentCoder().activeEditorId);
      currentCoder().activeEditorId = null;
    }

    currentCoder().dom.editorPane.innerHTML = '';
    currentCoder().activeFilePath = filePath;
    currentCoder().activeEditorFile = filePath;
    currentCoder().activeEditorCodeMindLanguage = _getCodeMindLanguage(filePath);
    // Save reads the LIVE active path, not this closure's captured
    // filePath — so a rename/move that retargets the open file (see
    // _retargetEditorPaths) saves to the new location, not the old one.
    // There's only ever one live editor (opening destroys the prior),
    // so the active-path globals always describe THIS editor.
    const saveCurrent = () => _saveFile(currentCoder().workspaceId, currentCoder().activeFilePath);
    const diagnosticsToken = ++currentCoder().activeEditorDiagnosticsToken;

    // Header + editor body must be separate flex children; mounting the
    // editor directly into coder-editor-pane and then inserting a header
    // above it makes the editor keep 100% height and clip at the bottom.
    const header = document.createElement('div');
    header.className = 'coder-editor-header';
    const pathParts = filePath.split('/').filter(Boolean);
    const breadcrumb = pathParts.map((p, i) =>
      i === pathParts.length - 1
        ? `<span class="coder-breadcrumb-active">${escapeHtml(p)}</span>`
        : `<span class="coder-breadcrumb-part">${escapeHtml(p)}</span>`
    ).join('<span class="coder-breadcrumb-sep">/</span>');
    header.innerHTML = `<div class="coder-breadcrumb">${breadcrumb}</div><div class="coder-editor-header-actions"><button class="coder-editor-diagnostics hidden" type="button" title=""></button><button class="coder-save-btn" title="Save (Ctrl+S)">Save</button></div>`;
    header.querySelector('.coder-editor-diagnostics').onclick = _focusFirstEditorDiagnostic;
    header.querySelector('.coder-save-btn').onclick = saveCurrent;

    const body = document.createElement('div');
    body.className = 'coder-editor-body';

    currentCoder().dom.editorPane.appendChild(header);
    currentCoder().dom.editorPane.appendChild(body);

    currentCoder().activeEditorId = Editor.create(body, {
      content,
      language,
      onSave: saveCurrent,
      onChange: () => { _markActiveDirty(); _queueEditorDiagnostics(); },
    });

    // Upsert the tab entry, carrying dirty state across the reopen and
    // caching the buffer so a later switch-back restores exactly this.
    let _tab = currentCoder().editorFiles.find((f) => f.path === filePath);
    if (!_tab) { _tab = { path: filePath, name: fileName }; currentCoder().editorFiles.push(_tab); }
    _tab.editorId = currentCoder().activeEditorId;
    _tab.dirty = _startDirty;
    _tab.buffer = content;

    // Show the editor split panel and render tabs
    _showEditorSplit();
    _renderEditorTabs();
    requestAnimationFrame(() => {
      // Jump-to-line (search result / diagnostic navigation). Position the
      // cursor at the match column so scrollIntoView centers the exact hit,
      // not just the top of the line. Deferred to the same frame as focus
      // so the editor's layout is measured before we scroll.
      if (jumpTo?.line && currentCoder().activeEditorId) {
        const col = Array.isArray(jumpTo.spans) && jumpTo.spans[0]
          ? (jumpTo.spans[0][0] || 0) + 1
          : 1;
        Editor.setCursor(currentCoder().activeEditorId, jumpTo.line, col);
      } else if (_restoreCursor && currentCoder().activeEditorId) {
        // Switch-back: land where the user last was in this file.
        Editor.setCursor(currentCoder().activeEditorId, _restoreCursor.line, _restoreCursor.col);
      } else {
        Editor.focus(currentCoder().activeEditorId);
      }
    });
    _primeEditorDiagnostics(content, filePath, diagnosticsToken);
    // Follow the active file in the tree (expand ancestors, highlight,
    // scroll into view). Fire-and-forget; cheap when already visible.
    _revealInTree(filePath);

  } catch (err) {
    console.error('Failed to open file:', err);
    showToast('Failed to open file', 'error');
  }
}

// Snapshot the live editor's buffer + cursor into its tab entry. Called
// before the editor is destroyed (tab switch / reopen) so unsaved edits
// and the cursor position survive the round-trip.
function _snapshotActiveEditorBuffer() {
  if (!currentCoder().activeEditorId || !currentCoder().activeEditorFile) return;
  const entry = currentCoder().editorFiles.find((f) => f.path === currentCoder().activeEditorFile);
  if (!entry) return;
  try {
    entry.buffer = Editor.getContent(currentCoder().activeEditorId);
    entry.cursor = Editor.getCursor(currentCoder().activeEditorId);
  } catch { /* editor already gone — nothing to snapshot */ }
}

// Flag the active tab dirty on first edit. Only re-renders on the
// clean→dirty transition, so it stays O(1) per keystroke.
function _markActiveDirty() {
  const entry = currentCoder().editorFiles.find((f) => f.path === currentCoder().activeEditorFile);
  if (entry && !entry.dirty) { entry.dirty = true; _renderEditorTabs(); }
}

async function _saveFile(workspaceId, filePath) {
  if (!currentCoder().activeEditorId) return;
  const content = Editor.getContent(currentCoder().activeEditorId);
  const savedPath = filePath || currentCoder().activeFilePath;
  try {
    const resp = await fetch(`/api/coder/files/${encodeURIComponent(workspaceId || currentCoder().workspaceId)}/write`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      // checkpoint:true stamps a "User edit: <path>" commit so manual
      // edits aren't silently rolled into the next agent's commit.
      body: JSON.stringify({ path: savedPath, content, checkpoint: true }),
    });
    if (resp.ok) {
      // Clear dirty + refresh the buffer baseline for this tab.
      const entry = currentCoder().editorFiles.find((f) => f.path === savedPath);
      if (entry) { entry.dirty = false; entry.buffer = content; }
      _renderEditorTabs();
      showToast('Saved', 'success');
      // If a new checkpoint was recorded, refresh the list so the tile
      // count and drawer stay fresh. `data.checkpoint` is null when
      // there was nothing to commit (idempotent save).
      try {
        const data = await resp.json();
        if (data?.checkpoint) _loadCheckpoints();
      } catch { /* response may have no body */ }
    } else {
      showToast('Save failed', 'error');
    }
  } catch { showToast('Save failed', 'error'); }
}

// ---------------------------------------------------------------------------
// Editor Tabs (in slide-in panel)
// ---------------------------------------------------------------------------

function _renderEditorTabs() {
  if (!currentCoder().dom.editorTabs) return;
  currentCoder().dom.editorTabs.innerHTML = currentCoder().editorFiles.map(f => {
    const active = f.path === currentCoder().activeEditorFile ? ' active' : '';
    const dirty = f.dirty ? ' is-dirty' : '';
    // A dirty tab shows a dot that becomes the \u00d7 close glyph on hover
    // (VS Code behavior). Both glyphs live in the DOM; CSS toggles them.
    return `<button class="coder-tab${active}${dirty}" data-path="${escapeHtml(f.path)}">
      <span class="coder-tab-name">${escapeHtml(f.name)}</span>
      <span class="coder-tab-close" title="Close">
        <span class="coder-tab-dot" aria-hidden="true">\u25cf</span>
        <span class="coder-tab-x" aria-hidden="true">\u00d7</span>
      </span>
    </button>`;
  }).join('');

  // Wire click handlers
  currentCoder().dom.editorTabs.querySelectorAll('.coder-tab').forEach(btn => {
    btn.addEventListener('click', (e) => {
      if (e.target.closest('.coder-tab-close')) {
        _closeEditorFile(btn.dataset.path);
      } else {
        const name = btn.querySelector('.coder-tab-name')?.textContent
          || btn.dataset.path.split('/').pop();
        _openFileInEditor(currentCoder().workspaceId, btn.dataset.path, name);
      }
    });
  });
}

function _closeEditorFile(filePath) {
  // Guard unsaved edits \u2014 the buffer only lives in memory.
  const closing = currentCoder().editorFiles.find((f) => f.path === filePath);
  if (closing?.dirty && !confirm(`Discard unsaved changes to ${closing.name}?`)) return;
  currentCoder().editorFiles = currentCoder().editorFiles.filter(f => f.path !== filePath);
  if (currentCoder().editorFiles.length === 0) {
    _hideEditorSplit();
    currentCoder().activeEditorFile = null;
    _resetEditorDiagnostics();
    if (currentCoder().activeEditorId) { Editor.destroy(currentCoder().activeEditorId); currentCoder().activeEditorId = null; }
  } else if (currentCoder().activeEditorFile === filePath) {
    // Switch to the last remaining file
    const last = currentCoder().editorFiles[currentCoder().editorFiles.length - 1];
    _openFileInEditor(currentCoder().workspaceId, last.path, last.name);
  }
  _renderEditorTabs();
}

// ---------------------------------------------------------------------------
// Status Line
// ---------------------------------------------------------------------------

function _updateStatus(state, detail = '') {
  if (currentCoder().dom.statusEl) {
    currentCoder().dom.statusEl.className = 'coder-status ' + state; // 'idle', 'executing', 'error'
    if (currentCoder().dom.statusText) currentCoder().dom.statusText.textContent = state;
    if (currentCoder().dom.statusDetail) currentCoder().dom.statusDetail.textContent = detail;
  }
  // Send button doubles as stop while the agent is executing.
  const sendBtn = document.getElementById('coder-send-btn');
  if (sendBtn) {
    if (state === 'executing') {
      sendBtn.classList.add('is-stop');
      sendBtn.title = 'Stop (Esc)';
      sendBtn.dataset.mode = 'stop';
    } else {
      sendBtn.classList.remove('is-stop');
      sendBtn.title = 'Send (Enter)';
      sendBtn.dataset.mode = 'send';
    }
  }
}

function _ensureRunDetailsButton() {
  if (!currentCoder().dom.statusEl || document.getElementById('coder-run-details-btn')) return;
  const btn = document.createElement('button');
  btn.id = 'coder-run-details-btn';
  btn.className = 'coder-run-details-btn';
  btn.type = 'button';
  btn.disabled = true;
  btn.textContent = 'Run Details';
  btn.addEventListener('click', () => _openRunDetailsDrawer());
  currentCoder().dom.statusEl.appendChild(btn);
}

function _setActiveRunDetails(runId) {
  if (!runId || runId === currentCoder().activeRunId) return;
  currentCoder().activeRunId = runId;
  _ensureRunDetailsButton();
  const btn = document.getElementById('coder-run-details-btn');
  if (btn) {
    btn.disabled = false;
    btn.textContent = `Run ${runId.slice(-6)}`;
    btn.title = 'Open Coder run details';
  }
  // Cooperative: surface the pause button now that there's something
  // to pause. Resets to the "running" label even if a prior pause
  // state from a different run leaked through.
  _setPauseButtonState('running');
  // Persist so a hard reload (mobile pull-to-refresh, browser
  // restore) can still find the in-flight run via /active-run on
  // the server side as soon as the workspace mounts. Keyed by
  // workspace so multiple tabs don't trample each other.
  try {
    if (currentCoder().workspaceId) {
      sessionStorage.setItem(`coder.activeRun.${currentCoder().workspaceId}`, runId);
    }
  } catch { /* sessionStorage disabled — ignore */ }
}

function _clearActiveRunDetails() {
  currentCoder().activeRunId = '';
  try {
    if (currentCoder().workspaceId) {
      sessionStorage.removeItem(`coder.activeRun.${currentCoder().workspaceId}`);
    }
  } catch { /* noop */ }
  const btn = document.getElementById('coder-run-details-btn');
  if (btn) {
    btn.disabled = true;
    btn.textContent = 'Run Details';
  }
  // Cooperative: pause button disappears + queue chip clears now
  // that the run is no longer reachable for interjection.
  _setPauseButtonState('hidden');
  _refreshQueueDepth(0);
}

/**
 * POST the cancel endpoint for the currently-active run.
 *
 * Background runs survive client disconnect, so aborting the fetch
 * alone no longer stops the agent — the user's Stop intent has to
 * go to the server. Best-effort; if the route 404s (run already
 * finished) or fails, we still abort locally so the UI unwinds.
 *
 * ``reason`` propagates into the handler's CancelledError path so
 * the next turn's ``<prior_turns>`` block can tell the model
 * whether the user pressed Stop, ran /clear, started a new turn,
 * etc. Canonical values: ``user_cancel``, ``slash_clear``,
 * ``slash_compact``, ``new_turn_started``, ``page_unload``.
 */
async function _cancelActiveRunOnServer(reason = 'user_cancel') {
  const runId = currentCoder().activeRunId;
  if (!runId) return;
  try {
    await fetch(`/api/coder/runs/${encodeURIComponent(runId)}/cancel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason }),
    });
  } catch { /* network glitch — broker.cancel is idempotent */ }
}

/**
 * Stop the active coder stream from this client.
 *
 * Sequence: ask the server to cancel the run (background runs
 * survive client disconnect by design), then commit the partial
 * assistant bubble with an explicit ``[cancelled: reason]`` marker
 * (so the model on the next turn reads a clean signal instead of
 * mid-sentence truncation), then abort the local fetch so the UI
 * unwinds immediately. All three calls are no-ops when nothing is
 * active.
 *
 * Use this in place of bare ``currentCoder().coderStream.abort()`` at every site
 * that represents a user-driven stop (Esc, Cancel button, /clear,
 * /compact, starting a new turn). Workspace switch / page unload
 * deliberately do NOT call this — those leave the run alone so a
 * mobile user walking into an elevator doesn't kill the run.
 *
 * @param {string} [reason='user_cancel'] Canonical cancellation
 *   reason — see ``_cancelActiveRunOnServer``. Echoed into the
 *   partial-bubble cleanup marker so the model has matching
 *   context in both the chat history and the next turn's
 *   ``<prior_turns>`` block.
 */
function _stopActiveCoderRun(reason = 'user_cancel') {
  if (currentCoder().coderStream?.isActive()) {
    void _cancelActiveRunOnServer(reason);
    // Commit the partial bubble BEFORE abort. Stream abort fires
    // onComplete('') which calls the regular finalizeResponse —
    // that path leaves the trailing partial text intact. By
    // calling the cancelled finalizer first, the streaming bubble
    // is replaced with a clean marker before the stream's own
    // cleanup runs (which then becomes a no-op since _streamEl is
    // already null).
    try { currentCoder().conversation?.finalizeResponseCancelled?.(reason); }
    catch { /* noop — DOM/markdown render shouldn't block cancel */ }
    try { currentCoder().coderStream.abort(); } catch { /* noop */ }
  }
}


/**
 * Show / hide the rewind button based on whether there's a turn to
 * rewind. Called after every state transition that might change the
 * conversation length: load, send, stream end, clear, workspace
 * switch. Cheap (one element class flip) so over-calling is fine.
 *
 * Visibility rule: button is visible when the conversation has at
 * least one user message AND we're not in the empty-onboarding
 * state. We don't gate on stream-active — rewind works mid-flight
 * too (cancels then restores) so the affordance stays available.
 */
function _refreshRewindAffordance() {
  const btn = document.getElementById('coder-rewind-btn');
  if (!btn) return;
  const messages = currentCoder().conversation?.getHistory?.() || [];
  const hasUser = messages.some(m => m.role === 'user');
  btn.classList.toggle('hidden', !hasUser);
}


/**
 * Send a cooperative interjection (queue or steer) into the active
 * run's inbox. Drains the composer textarea + attachments the same
 * way ``_sendConversationMessage`` does for a new turn, but posts to
 * the /interject endpoint instead of starting a new turn.
 *
 * ``mode``:
 *  - ``"steer"`` (default 2026-05-31) drains at the next iteration
 *    boundary inside the current turn. Appended as a user message so
 *    the model sees the redirect at the top of its next backend
 *    round-trip — typically within seconds. Use when you want to
 *    course-correct mid-turn before intent drifts.
 *  - ``"queue"`` drains at end-of-turn and chains as a new turn.
 *    Use when you have a clean follow-up that should wait for the
 *    current work to finish.
 *
 * Renders the typed content as a regular user message bubble with a
 * "queued" badge so the user has visual feedback of what's pending.
 * The badge flips to "delivered" when a subsequent stream chunk
 * (``status='steer_delivered'`` or ``status='queue_followup'``)
 * confirms the model has consumed it.
 *
 * Best-effort: HTTP failure shows an error toast and leaves the
 * composer content intact so the user can retry.
 */
async function _sendInterjection(mode = 'steer') {
  const runId = currentCoder().activeRunId;
  if (!runId) {
    showToast('No active run to interject into.', 'error');
    return;
  }
  const input = document.getElementById('coder-input');
  if (!input) return;
  const text = input.value.trim();
  const attachments = _drainAttachments();
  if (!text && attachments.length === 0) return;

  // Clear composer immediately so the user sees the send "land".
  input.value = '';
  input.style.height = 'auto';

  // Render in chat with a "queued" badge — the badge flips to
  // "delivered" on the matching stream chunk.
  const localMsgId = currentCoder().conversation?.addUserMessage(text, attachments) || '';
  if (localMsgId) {
    _markConversationMessageAsQueued(localMsgId, mode);
  }
  _refreshRewindAffordance();

  // POST the interjection. Server returns a msg_id we use to
  // correlate with the eventual delivery acknowledgment.
  try {
    const resp = await fetch(
      `/api/coder/runs/${encodeURIComponent(runId)}/interject`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: text,
          attachments,
          mode,
        }),
      },
    );
    if (!resp.ok) {
      // 409 specifically means the run finished mid-fetch (race we
      // can't avoid) — fall back to dispatching as a new turn instead
      // of leaving the user with a "queued" pill that will never
      // deliver. This is what they wanted anyway: the message + a
      // run to consume it. Bubble #1 is already rendered with the
      // text; mark it "delivered" (since it WAS accepted by the user's
      // intent, just at a moment with no run to receive it) and
      // dispatch a fresh turn that REUSES this bubble — guarded by
      // _suppressNextUserBubble so _runAgentInConversation skips its
      // own addUserMessage and we don't get the duplicate.
      if (resp.status === 409) {
        if (localMsgId) {
          _markConversationMessageAsQueued(localMsgId, 'queue');
          const container = document.getElementById('coder-conv-messages');
          const safeId = String(localMsgId).replace(/"/g, '\\"');
          const bubble = container?.querySelector(
            `.coder-msg-user[data-conv-msg-id="${safeId}"]`,
          );
          if (bubble) {
            bubble.dataset.coopState = 'delivered';
            const badge = bubble.querySelector('.coder-msg-coop-badge');
            if (badge) badge.textContent = 'delivered';
          }
        }
        _suppressNextUserBubble = true;
        _runAgentInConversation(text, attachments);
        return;
      }
      const data = await resp.json().catch(() => ({}));
      showToast(
        `Interjection rejected: ${data.error || resp.statusText}`,
        'error',
      );
      // Roll the user message back to "failed" rather than leaving
      // a queued chip that will never deliver.
      if (localMsgId) {
        _markConversationMessageAsQueued(localMsgId, 'failed');
      }
      return;
    }
    const data = await resp.json();
    _refreshQueueDepth(data.queue_depth || 0);
  } catch (exc) {
    showToast(`Interjection failed: ${exc?.message || exc}`, 'error');
  }
}


function _markConversationMessageAsQueued(msgId, mode) {
  // Find the rendered bubble by the conv-msg-id stamped on it at
  // creation time (addUserMessage in coder-conversation.js). Pre-fix
  // this used `container.lastElementChild` which was correct at
  // first-tag time (synchronous after addUserMessage) but WRONG when
  // re-tagging after a fetch await: by the time a 409 fallback called
  // this with mode='failed', the last child was usually an assistant
  // streaming chunk OR the auto-chained turn's duplicate user bubble,
  // so the "failed" badge silently landed on the wrong message and
  // the original "queued" bubble stayed queued forever.
  const container = document.getElementById('coder-conv-messages');
  if (!container || msgId == null) return;
  const safeId = String(msgId).replace(/"/g, '\\"');
  const bubble = container.querySelector(
    `.coder-msg-user[data-conv-msg-id="${safeId}"]`,
  );
  if (!bubble) return;
  bubble.dataset.coopState = mode === 'failed' ? 'failed' : 'queued';
  bubble.dataset.coopMode = mode;
  let badge = bubble.querySelector('.coder-msg-coop-badge');
  if (!badge) {
    badge = document.createElement('span');
    badge.className = 'coder-msg-coop-badge';
    bubble.appendChild(badge);
  }
  badge.textContent = mode === 'failed' ? 'failed'
    : mode === 'steer' ? 'steering'
    : 'queued';
  bubble.dataset.coopMsgId = msgId;
}


function _refreshQueueDepth(depth) {
  // The depth chip lives next to the rewind button. Hide when 0.
  const chip = document.getElementById('coder-queue-chip');
  if (!chip) return;
  if (!depth) {
    chip.classList.add('hidden');
    chip.textContent = '';
    return;
  }
  chip.classList.remove('hidden');
  chip.textContent = depth === 1 ? '1 queued' : `${depth} queued`;
}


async function _togglePauseActiveRun() {
  const runId = currentCoder().activeRunId;
  if (!runId) return;
  // The button's data-state attribute tracks paused vs running so a
  // single click handler can dispatch to the right endpoint without
  // racing the server's view.
  const btn = document.getElementById('coder-pause-btn');
  const isPaused = btn?.dataset.state === 'paused';
  const endpoint = isPaused ? 'resume' : 'pause';
  try {
    const resp = await fetch(
      `/api/coder/runs/${encodeURIComponent(runId)}/${endpoint}`,
      { method: 'POST' },
    );
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      showToast(
        `${endpoint}: ${data.reason || resp.statusText}`,
        'warning',
      );
      return;
    }
    // Flip the button state. The stream will keep emitting (or
    // resume emitting after a pause) — no UI refresh needed
    // beyond the button label.
    _setPauseButtonState(isPaused ? 'running' : 'paused');
  } catch (exc) {
    showToast(`${endpoint} failed: ${exc?.message || exc}`, 'error');
  }
}


function _setPauseButtonState(state) {
  // Visibility + label state for the pause/resume toggle. Hidden
  // when no run is active so the composer doesn't have a useless
  // button. Resume vs Pause label flips so the same button serves
  // both states without users hunting for the right control.
  const btn = document.getElementById('coder-pause-btn');
  if (!btn) return;
  if (state === 'hidden' || !currentCoder().activeRunId) {
    btn.classList.add('hidden');
    btn.dataset.state = '';
    btn.title = '';
    return;
  }
  btn.classList.remove('hidden');
  btn.dataset.state = state;
  if (state === 'paused') {
    btn.title = 'Resume the agent';
    btn.textContent = '▶';
  } else {
    btn.title = 'Pause the agent at the next iteration boundary';
    btn.textContent = '⏸';
  }
}


/**
 * Rewind the last coder turn.
 *
 * Sequence:
 *   1. Confirm with the user — rewind is destructive (workspace
 *      files change, message bubbles disappear).
 *   2. POST /api/coder/workspaces/{id}/rewind. Backend cancels any
 *      in-flight run, restores files from the TurnSnapshot, pops
 *      the matching turn_summary, clears per-request scratchpads.
 *   3. Abort the local stream if running (the server cancel was
 *      already requested; this collapses the UI to idle without
 *      waiting for the broker subscription to drain).
 *   4. Drop the matching user→assistant pair from the conversation
 *      DOM + history via ``rewindLastTurn()``.
 *   5. Save the cleaned conversation tree back to the server.
 *   6. Refresh the file tree so the user sees restored content.
 *   7. Show a toast summarising what happened (paths restored,
 *      irreversible side effects, etc.).
 *
 * Best-effort throughout — a failure at any step shows an error
 * toast but doesn't roll back earlier steps. The backend response's
 * ``warnings`` array is shown to the user verbatim.
 */
function _showRewindMenu(event) {
  // Convergence-style 3-mode rewind picker. CC's vocabulary:
  //   both  = workspace + conversation (the original rewind)
  //   files = restore files only, keep the chat history
  //   conv  = drop conversation + state, keep edits on disk
  //           (the "poisoned context" cure)
  // Implemented as a native <select>-driven prompt for v1 — small,
  // accessible, doesn't require a popover library. A bespoke
  // floating menu is a polish PR.
  const choice = window.prompt(
    'Rewind mode:\n'
    + '  both  — undo files AND remove the message exchange\n'
    + '  files — undo files only (keep the chat)\n'
    + '  conv  — drop the message exchange only (keep edits)\n\n'
    + 'Type one of: both / files / conv',
    'both',
  );
  if (!choice) return;
  const mode = String(choice).trim().toLowerCase();
  if (!['both', 'files', 'conv'].includes(mode)) {
    showToast(`Unknown rewind mode: ${choice}`, 'error');
    return;
  }
  void _rewindLastCoderTurn(mode);
}


async function _rewindLastCoderTurn(mode = 'both') {
  const workspaceId = currentCoder().workspaceId;
  if (!workspaceId) {
    showToast('No active workspace.', 'error');
    return;
  }
  const confirmText = {
    both: 'Rewind the last turn?\n\nThis will undo file changes AND remove the last message exchange.',
    files: 'Rewind workspace files for the last turn?\n\nThe conversation history will be kept; only files are restored.',
    conv: 'Drop the last message exchange?\n\nFile edits stay on disk — only the conversation + agent state are rewound.',
  }[mode] || 'Rewind?';
  const sideEffectWarning = mode === 'conv' ? '' :
    '\n\nSide effects outside the workspace (HTTP requests, started services, git pushes) will not be undone.';
  if (!confirm(confirmText + sideEffectWarning)) return;

  const rewindBtn = document.getElementById('coder-rewind-btn');
  if (rewindBtn) rewindBtn.disabled = true;

  let outcome;
  try {
    const resp = await fetch(
      `/api/coder/workspaces/${encodeURIComponent(workspaceId)}/rewind`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      },
    );
    outcome = await resp.json();
    if (!resp.ok || !outcome?.ok) {
      const msg = outcome?.error || `Rewind failed (HTTP ${resp.status}).`;
      showToast(msg, 'error');
      return;
    }
  } catch (exc) {
    showToast(`Rewind request failed: ${exc?.message || exc}`, 'error');
    return;
  } finally {
    if (rewindBtn) rewindBtn.disabled = false;
  }

  // Abort the local stream if one was active (server already
  // requested cancel; this just unwinds the UI).
  if (currentCoder().coderStream?.isActive()) {
    try { currentCoder().coderStream.abort(); } catch { /* noop */ }
    _updateStatus('idle');
  }

  // Drop the last user→assistant exchange from the conversation —
  // ONLY for modes that touch conversation/state. ``files`` mode
  // explicitly keeps the chat so the model remembers the discussion
  // even though the edits got rolled back.
  let removed = 0;
  if (mode === 'both' || mode === 'conv') {
    removed = currentCoder().conversation?.rewindLastTurn?.() || 0;
    if (removed > 0) {
      await _saveConversation(workspaceId);
    }
  }
  _refreshRewindAffordance();

  // Refresh the file tree so restored content is visible — same
  // event the review-reject path fires. Skip for ``conv`` since
  // files weren't touched.
  if (mode !== 'conv') {
    window.dispatchEvent(new CustomEvent('coder:turn-reviewed', {
      detail: { workspaceId, kind: 'rewound', runId: outcome.run_id || '', mode },
    }));
  }

  // Toast summary scoped to the mode so the message matches what
  // actually happened. The backend's warnings list is shown verbatim
  // so the user sees the side-effect caveats.
  const restoredCount = (outcome.restored_paths || []).length;
  const irrevCount = (outcome.irreversible_paths || []).length;
  const bits = [];
  if (removed > 0) bits.push(`Removed ${removed} message${removed === 1 ? '' : 's'}.`);
  if (restoredCount) bits.push(`${restoredCount} file${restoredCount === 1 ? '' : 's'} restored.`);
  if (irrevCount) bits.push(`${irrevCount} non-reversible.`);
  if (!bits.length) bits.push(`Rewind (${mode}) complete.`);
  showToast(bits.join(' '), irrevCount ? 'warning' : 'success');
  if (Array.isArray(outcome.warnings)) {
    for (const w of outcome.warnings) {
      if (w) showToast(w, 'info');
    }
  }
}


// ── Planning-mode cycle ────────────────────────────────────────────
// Auto → Approve → Plan → Auto. ``auto`` is the default (zero
// friction — model runs freely). Approve adds per-tool permission
// prompts on mutations. Plan adds a "propose first" system-prompt
// nudge while keeping all tools available (the hard tool filter was
// retired in migration 208 — soft guidance, not enforcement).
//
// Click on the badge OR Shift+Tab in the textarea cycles. Persists
// per-workspace. New mode takes effect on the NEXT turn (already-
// running turns finish under their start mode so a Shift+Tab mid-
// flight doesn't yank the rug).
//
// Backend values stay the legacy strings (``default``, ``plan``,
// ``auto``) for back-compat with persisted rows + migration history.
// Frontend translates ``default`` → "approve" for display clarity.
const _PLANNING_MODE_CYCLE = ['auto', 'default', 'plan'];
const _PLANNING_MODE_LABEL = {
  auto: 'auto',
  default: 'approve',
  plan: 'plan',
};

async function _cyclePlanningMode() {
  const workspaceId = currentCoder().workspaceId;
  if (!workspaceId) {
    showToast('No active workspace.', 'error');
    return;
  }
  const btn = document.getElementById('coder-plan-mode-btn');
  const current = btn?.dataset.mode || 'default';
  const idx = _PLANNING_MODE_CYCLE.indexOf(current);
  const next = _PLANNING_MODE_CYCLE[(idx + 1) % _PLANNING_MODE_CYCLE.length];
  // Update UI optimistically; revert on HTTP failure.
  _setPlanningModeBadge(next);
  try {
    const resp = await fetch(
      `/api/coder/workspaces/${encodeURIComponent(workspaceId)}/planning-mode`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: next }),
      },
    );
    if (!resp.ok) {
      _setPlanningModeBadge(current);
      const data = await resp.json().catch(() => ({}));
      showToast(`Planning-mode change failed: ${extractErrorMessage(data, resp.statusText)}`, 'error');
      return;
    }
    showToast(`Planning mode: ${next}`, 'info');
  } catch (exc) {
    _setPlanningModeBadge(current);
    showToast(`Planning-mode request failed: ${exc?.message || exc}`, 'error');
  }
}


function _setPlanningModeBadge(mode) {
  const btn = document.getElementById('coder-plan-mode-btn');
  if (!btn) return;
  const safe = _PLANNING_MODE_CYCLE.includes(mode) ? mode : 'auto';
  btn.dataset.mode = safe;
  const label = btn.querySelector('.coder-plan-mode-label');
  if (label) label.textContent = _PLANNING_MODE_LABEL[safe] || safe;
}


async function _refreshPlanningModeBadge(workspaceId) {
  // Reads the workspace info via the existing list endpoint and
  // updates the composer's plan-mode badge to match. Fire-and-
  // forget; failure silently leaves the badge at its current state
  // (which defaults to "default" on initial render).
  if (!workspaceId) return;
  try {
    const resp = await fetch('/api/coder/workspaces');
    if (!resp.ok) return;
    const data = await resp.json();
    const ws = (data.workspaces || data || []).find?.(w => w.id === workspaceId);
    if (ws && ws.planning_mode) {
      _setPlanningModeBadge(ws.planning_mode);
    } else {
      _setPlanningModeBadge('auto');
    }
  } catch {
    /* network glitch — badge stays where it is */
  }
}


// ── Thinking toggle ───────────────────────────────────────────────
// Per-workspace per-turn override for the model's enable_thinking
// chat-template kwarg. Hidden when the active model doesn't support
// the kwarg (non-Qwen/GLM/EXAONE/Nemotron). State persists across
// page reloads via localStorage so the user's choice survives the
// "turn things off before execute" workflow even if they refresh.

function _coderThinkingStorageKey(workspaceId) {
  return `coder.thinking.${workspaceId || 'default'}`;
}


function _isCoderThinkingEnabled() {
  // Source of truth = the button's data-active attribute. We mirror
  // localStorage at write time so the next page load can restore it
  // without a server round-trip.
  const btn = document.getElementById('coder-thinking-btn');
  return btn?.dataset.active === 'true';
}


function _refreshCoderThinkingToggle() {
  // Show/hide based on the current model's capability. Restore the
  // persisted active state from localStorage if it's available.
  // Called on: init, workspace switch, model change.
  const btn = document.getElementById('coder-thinking-btn');
  if (!btn) return;
  const model = app.state.currentModel || '';
  const supported = supportsThinkingToggleForModel(model);
  btn.classList.toggle('hidden', !supported);
  if (!supported) {
    // Force off when hidden so a model swap doesn't silently leave
    // an irrelevant toggle "active" in the persisted state.
    btn.dataset.active = 'false';
    btn.setAttribute('aria-pressed', 'false');
    return;
  }
  // Restore persisted state for the current workspace.
  const key = _coderThinkingStorageKey(currentCoder().workspaceId);
  let stored = 'false';
  try { stored = localStorage.getItem(key) || 'false'; } catch { /* private mode */ }
  btn.dataset.active = stored === 'true' ? 'true' : 'false';
  btn.setAttribute('aria-pressed', btn.dataset.active);
}


function _toggleCoderThinking() {
  const btn = document.getElementById('coder-thinking-btn');
  if (!btn || btn.classList.contains('hidden')) return;
  const next = btn.dataset.active === 'true' ? 'false' : 'true';
  btn.dataset.active = next;
  btn.setAttribute('aria-pressed', next);
  // Persist immediately so a refresh mid-workflow doesn't lose the
  // user's choice (the canonical "toggle on for plan turn, off for
  // execute turn" pattern survives page reloads + tab restores).
  try {
    localStorage.setItem(
      _coderThinkingStorageKey(currentCoder().workspaceId),
      next,
    );
  } catch { /* private-browsing — toggle still works in-memory */ }
  showToast(
    next === 'true'
      ? 'Coder thinking: ON (model will reason before tool calls this turn)'
      : 'Coder thinking: OFF (faster tool-calling, no chain-of-thought)',
    'info',
  );
}


// ── Cooperative stream-chunk handler ──────────────────────────────
// Called from coder-stream.js when a recognized cooperative aug
// status comes through. Handles three events today:
//   * status="steer_delivered"  — flip queued/steering badges on
//                                 user-message bubbles to delivered.
//   * status="queue_followup"   — at end-of-turn, the broker drained
//                                 the queue inbox; auto-chain a new
//                                 turn with the queued content as
//                                 the user prompt. DEFERRED to
//                                 onComplete via ``_pendingCoopChain``
//                                 because the stream is still
//                                 ``isActive()`` at this moment and
//                                 firing send-conversation directly
//                                 would re-queue the message in an
//                                 infinite ping-pong.
//   * status="queue_dropped"    — cancel/error path drained the inbox;
//                                 flip queued bubbles to "dropped".

// Pending queue-followup chain. queue_followup arrives WHILE the
// stream is still draining (in the onStatus callback). Firing
// ``_sendConversationMessage`` directly would see the still-active
// stream and re-queue the message into the inbox — infinite ping-
// pong, message stays queued forever (live regression observed
// 2026-05-30). Fix: stash the drained payload here, dispatch it
// from onComplete after the stream has actually closed.
let _pendingCoopChain = null;

// When set, the next _runAgentInConversation call skips its own
// addUserMessage step because a bubble for this message is already
// rendered (from the original interject path that got 409, or from
// the queued-then-drained coop bubble flipped to 'delivered'). Without
// this, the auto-chain or 409-fallback dispatches re-render the same
// user content as a SECOND bubble — the user-reported duplicate.
let _suppressNextUserBubble = false;

// Dedup set of queue_followup chain keys we've already auto-chained.
// Key shape: ``<runId>:<msgId>`` so a fresh interjection that happens
// to reuse a msgId from a previous run can't collide. Guards the
// reconnect / replay case where a finished broker run's buffer is
// re-streamed — without this, the same chain would fire again as a
// new turn on every reconnect. Bounded to keep memory flat across
// long sessions.
const _seenCoopChainIds = new Set();
const _SEEN_COOP_CHAIN_CAP = 64;

function _coopChainKey(runId, msgId) {
  return `${String(runId || '')}:${String(msgId || '')}`;
}


function _handleCoopChunk(aug) {
  // Defensive at the call boundary so the onStatus callback doesn't
  // need to wrap us in a try/catch — every DOM operation below is
  // null-guarded, and unrecognized statuses bail before touching
  // anything. The only exception surface left is a corrupted
  // ``aug.messages`` payload (handled explicitly via Array.isArray).
  if (!aug || typeof aug !== 'object') return;
  const status = aug.status;
  if (
    status !== 'steer_delivered'
    && status !== 'queue_followup'
    && status !== 'queue_dropped'
  ) return;

  if (status === 'steer_delivered') {
    _flipCoopBubblesToDelivered();
    _refreshQueueDepth(0);
    return;
  }
  if (status === 'queue_dropped') {
    // Cancel / error path drained the inbox. Flip every queued
    // bubble to "dropped" so the user sees the message wasn't
    // delivered and can retype. Distinct from "failed" (HTTP error
    // on the enqueue itself) — "dropped" means it was accepted into
    // the inbox but the owning turn was interrupted before drain.
    _flipCoopBubblesToDropped(aug.reason || 'cancelled');
    _refreshQueueDepth(0);
    // Also clear any pending chain — dropped means the user's
    // queued content is gone; don't auto-chain it as a new turn.
    _pendingCoopChain = null;
    return;
  }
  // status === 'queue_followup' — stash for onComplete to dispatch.
  // The bubbles are safe to flip + queue depth safe to clear now
  // because the backend has already drained the inbox at this point.
  _flipCoopBubblesToDelivered();
  _refreshQueueDepth(0);
  const msgs = Array.isArray(aug.messages) ? aug.messages : [];
  if (!msgs.length) return;

  // Run id used for the dedup key. Falls back to '' which still works
  // (the msg_id is server-generated and reasonably unique on its own;
  // the run-id qualifier is just defence-in-depth against cross-run
  // collisions).
  const runId = String(currentCoder().activeRunId || '');

  // The chained turn dispatches msgs[0]. If the user queued 2+ messages
  // while the turn ran, we surface a count so they aren't silently
  // dropped — pre-fix they were. Future improvement: drain them all in
  // sequence, but that requires backend cooperation to suppress the
  // next end-of-turn drain. For now, telling the user "N more were
  // queued, resend them" is honest and avoids the silent-loss bug.
  if (msgs.length > 1) {
    const extra = msgs.length - 1;
    showToast(
      `${extra} additional queued message${extra === 1 ? '' : 's'} weren't auto-chained — resend if still needed.`,
      'warning',
    );
    console.warn(
      "[coder.coop] queue_followup carried multiple messages; only the first auto-chains",
      { total: msgs.length, dispatched: 0, deferred: extra },
    );
  }

  const next = msgs[0];
  if (!next || typeof next !== 'object') return;
  const content = String(next.content || '');
  if (!content) return;
  // Dedup — reconnect/replay would otherwise fire the chain again
  // for the same drained inbox entry. Keyed by (runId, msgId) so a
  // fresh interjection on a NEW run with a reused msg_id can't be
  // silently swallowed by an earlier run's chain.
  const msgId = String(next.id || '');
  const chainKey = _coopChainKey(runId, msgId);
  if (msgId && _seenCoopChainIds.has(chainKey)) {
    console.log("[coder.coop] queue_followup already chained for key, skipping:", chainKey);
    return;
  }
  if (msgId) {
    _seenCoopChainIds.add(chainKey);
    if (_seenCoopChainIds.size > _SEEN_COOP_CHAIN_CAP) {
      // Evict oldest entry (insertion order is preserved in Set).
      const oldest = _seenCoopChainIds.values().next().value;
      _seenCoopChainIds.delete(oldest);
    }
  }
  // Stash for onComplete. If two queue_followup chunks arrive in
  // one turn (shouldn't, but defensive), the LATEST wins — that's
  // the freshest user intent.
  _pendingCoopChain = {
    msgId,
    content,
    attachments: Array.isArray(next.attachments) ? next.attachments : [],
  };
  console.log(
    "[coder.coop] queue_followup stashed for onComplete dispatch",
    { msgId, contentPreview: content.slice(0, 80), attachments: _pendingCoopChain.attachments.length },
  );
}


function _flushPendingCoopChain() {
  // Called from onComplete once the stream has fully closed. At
  // this point ``currentCoder().coderStream.isActive()`` returns false, so
  // ``_sendConversationMessage`` will route to the new-turn path
  // instead of the cooperative-interjection-back-into-inbox path.
  // Without this defer-then-flush dance, the chained turn would
  // ping-pong back into the inbox forever (live bug 2026-05-30).
  if (!_pendingCoopChain) return;
  const chain = _pendingCoopChain;
  _pendingCoopChain = null;
  const input = document.getElementById('coder-input');
  if (!input) return;
  input.value = chain.content;
  if (chain.attachments.length) {
    currentCoder().pendingAttachments.push(...chain.attachments);
  }
  console.log(
    "[coder.coop] dispatching chained turn from queue_followup:",
    chain.content.slice(0, 80),
  );
  // The original interject already rendered a user-message bubble
  // (now flipped to "delivered" via _flipCoopBubblesToDelivered).
  // Suppress the next addUserMessage so the chained turn doesn't
  // create a SECOND bubble with the same content — that was the
  // visible duplicate users reported.
  _suppressNextUserBubble = true;
  _sendConversationMessage({ mode: 'auto' });
}


function _flipCoopBubblesToDelivered() {
  // Shared between steer_delivered and queue_followup handlers —
  // flips every still-queued user-message bubble to "delivered"
  // (badge + dataset). Null-guarded so missing container or no
  // queued bubbles is a quiet no-op.
  const container = document.getElementById('coder-conv-messages');
  if (!container) return;
  const queued = container.querySelectorAll(
    '.coder-msg-user[data-coop-state="queued"]',
  );
  queued.forEach(el => {
    el.dataset.coopState = 'delivered';
    const badge = el.querySelector('.coder-msg-coop-badge');
    if (badge) badge.textContent = 'delivered';
  });
}


function _flipCoopBubblesToDropped(reason) {
  // Cancel/error path: surface dropped queued messages so the user
  // knows the turn was interrupted before drain and they need to
  // retype. The dropped badge styling reuses the "failed" red so
  // the visual distinction from "queued" (accent color) and
  // "delivered" (green) is immediate.
  const container = document.getElementById('coder-conv-messages');
  if (!container) return;
  const queued = container.querySelectorAll(
    '.coder-msg-user[data-coop-state="queued"]',
  );
  if (!queued.length) return;
  queued.forEach(el => {
    el.dataset.coopState = 'dropped';
    const badge = el.querySelector('.coder-msg-coop-badge');
    if (badge) {
      badge.textContent = 'dropped';
      badge.title = `Turn interrupted (${reason || 'cancelled'}) — retype to send`;
    }
  });
}


function _sweepStaleCoopBubbles(reason) {
  // Missed-chunk safety net, called after the stream has fully closed
  // (onComplete / onError). Every inbox entry is resolved server-side
  // at turn exit — steer_delivered at an iteration boundary,
  // queue_followup at natural completion (which since 2026-07-03 also
  // promotes undelivered steers), queue_dropped on cancel/error. So a
  // bubble STILL in coop-state="queued" here means its resolution
  // chunk never reached us (disconnect mid-cancel, replay gap). Before
  // this sweep, that bubble kept its "steering"/"queued" badge forever
  // and the user had no signal the message was lost.
  const container = document.getElementById('coder-conv-messages');
  if (!container) return;
  const stale = container.querySelectorAll(
    '.coder-msg-user[data-coop-state="queued"]',
  );
  if (!stale.length) return;
  console.warn(
    '[coder.coop] stream ended with unresolved queued/steer bubbles — flipping to dropped',
    { count: stale.length, reason },
  );
  _flipCoopBubblesToDropped(reason || 'turn_ended');
  _refreshQueueDepth(0);
  showToast(
    'A message sent mid-turn was never delivered — resend if still needed.',
    'warning',
  );
}

/**
 * On workspace activation, ask the server if a run is still in
 * flight for this workspace. If so, attach the CoderStream to the
 * /stream endpoint so the user sees progress without re-sending the
 * prompt. No-op when the workspace is idle.
 */
async function _attemptCoderReconnect(workspaceId) {
  if (!workspaceId) return;
  let active;
  try {
    const resp = await fetch(
      `/api/coder/workspaces/${encodeURIComponent(workspaceId)}/active-run`,
    );
    if (!resp.ok) return;
    active = await resp.json();
  } catch {
    return;
  }
  const runId = active?.run_id;
  if (!runId) return;
  // Build a fresh stream and reattach. _runAgentInConversation will
  // wire its own callbacks on the next user prompt — this attach
  // path only fires for the reconnect.
  if (currentCoder().coderStream?.isActive()) return;
  _setActiveRunDetails(runId);
  _updateStatus('executing', 'reconnecting...');
  currentCoder().coderStream = new CoderStream(_buildCoderStreamCallbacks());
  try {
    // Resume past what we already rendered for this run (persisted at
    // status boundaries), so a repeat reconnect or a page reload tails
    // new events instead of replaying the whole ring buffer from 0.
    await currentCoder().coderStream.attach({ runId, sinceSeq: readRunSeq(runId) });
  } catch (err) {
    console.warn('coder reattach failed', err);
  }
}

async function _openRunDetailsDrawer() {
  if (!currentCoder().activeRunId) return;
  try {
    const [runResp, eventsResp] = await Promise.all([
      fetch(`/api/coder/runs/${encodeURIComponent(currentCoder().activeRunId)}`),
      fetch(`/api/coder/runs/${encodeURIComponent(currentCoder().activeRunId)}/events?limit=120`),
    ]);
    if (!runResp.ok) throw new Error(await runResp.text());
    const run = (await runResp.json()).run || {};
    const events = eventsResp.ok ? ((await eventsResp.json()).events || []) : [];
    _renderRunDetailsDrawer(run, events);
  } catch (err) {
    showToast?.(`Run details unavailable: ${err.message || err}`, 'error');
  }
}

function _renderRunDetailsDrawer(run, events) {
  if (!currentCoder().runDetailsDrawer) {
    currentCoder().runDetailsDrawer = document.createElement('aside');
    currentCoder().runDetailsDrawer.className = 'coder-run-details-drawer';
    document.body.appendChild(currentCoder().runDetailsDrawer);
  }
  const metrics = run.metrics_json || {};
  const closeout = run.closeout_json || {};
  const list = (items) => (Array.isArray(items) && items.length)
    ? items.map((x) => `<li>${escapeHtml(String(x))}</li>`).join('')
    : '<li class="muted">None recorded</li>';
  const eventRows = events.slice(-60).map((ev) => `
    <tr>
      <td>${escapeHtml(String(ev.seq || ''))}</td>
      <td>${escapeHtml(ev.phase || '')}</td>
      <td>${escapeHtml(ev.status || ev.type || '')}</td>
    </tr>
  `).join('');
  const tokenRows = Array.isArray(metrics.token_snapshots)
    ? metrics.token_snapshots.slice(-8).map((snap) => {
        const scope = snap.scope || 'prompt';
        const tokens = _formatTokenCount(snap.tokens);
        const limit = _formatTokenCount(snap.limit);
        const suffix = snap.compacted ? ' after compact' : '';
        return `<li>${escapeHtml(scope)}: ${tokens}/${limit} tokens${suffix}</li>`;
      }).join('')
    : '';
  currentCoder().runDetailsDrawer.innerHTML = `
    <div class="coder-run-details-head">
      <div>
        <h2>Run Details</h2>
        <p>${escapeHtml(run.id || currentCoder().activeRunId)}</p>
      </div>
      <button type="button" class="coder-run-details-close" aria-label="Close">×</button>
    </div>
    <div class="coder-run-details-grid">
      <div><span>Strategy</span><strong>${escapeHtml(run.strategy || 'unknown')}</strong></div>
      <div><span>Model</span><strong>${escapeHtml(run.model || 'unknown')}</strong></div>
      <div><span>Status</span><strong>${escapeHtml(run.status || 'unknown')}</strong></div>
      <div><span>Tools</span><strong>${escapeHtml(String(run.tool_calls || 0))}</strong></div>
      <div><span>Iterations</span><strong>${escapeHtml(String(run.iterations || 0))}</strong></div>
      <div><span>Max Tokens</span><strong>${escapeHtml(_formatTokenCount(metrics.max_prompt_tokens))}</strong></div>
      <div><span>Compactions</span><strong>${escapeHtml(String(metrics.compactions || 0))}</strong></div>
      <div><span>Visible Answer</span><strong>${metrics.visible_answer ? 'yes' : 'no'}</strong></div>
      <div><span>Verification</span><strong>${metrics.verification_coverage ? 'covered' : 'not recorded'}</strong></div>
      <div><span>Checkpoint</span><strong>${escapeHtml(run.checkpoint_id || 'none')}</strong></div>
    </div>
    <section>
      <h3>Token Budget</h3>
      <ul>${tokenRows || '<li class="muted">None recorded</li>'}</ul>
    </section>
    <section>
      <h3>Closeout</h3>
      <ul>${list(closeout.changed_files || run.changed_files)}</ul>
    </section>
    <section>
      <h3>Commands</h3>
      <ul>${list(run.commands_run)}</ul>
    </section>
    <section>
      <h3>Tests And Browser Checks</h3>
      <ul>${list([...(run.tests_run || []), ...(run.browser_checks || [])])}</ul>
    </section>
    <section>
      <h3>Recent Events</h3>
      <table>
        <thead><tr><th>#</th><th>Phase</th><th>Status</th></tr></thead>
        <tbody>${eventRows || '<tr><td colspan="3">No events recorded</td></tr>'}</tbody>
      </table>
    </section>
  `;
  currentCoder().runDetailsDrawer.querySelector('.coder-run-details-close')
    ?.addEventListener('click', () => currentCoder().runDetailsDrawer.classList.remove('open'));
  requestAnimationFrame(() => currentCoder().runDetailsDrawer.classList.add('open'));
}

// ---------------------------------------------------------------------------
// Intent Bar
// ---------------------------------------------------------------------------

const _INTENT_PATTERNS = [
  // Test/syntax errors — fix is the obvious action
  { match: /SyntaxError|IndentationError/i, suggestions: [
    {label: 'Fix syntax error', action: '//fix the syntax error in the output above'},
    {label: 'Read the file', action: '//read the file mentioned in the error'},
    {label: 'Undo last edit', action: '//revert the last file change'},
    {label: 'Explain', action: '//explain what went wrong'},
  ]},
  { match: /FAIL|FAILED|AssertionError|assert.*Error/i, suggestions: [
    {label: 'Fix failing test', action: '//fix the failing test'},
    {label: 'Show full error', action: '//show the complete error with context'},
    {label: 'Run tests verbose', action: 'python3 -m pytest -v'},
    {label: 'Undo', action: '//revert the last change'},
  ]},
  { match: /TypeError|NameError|ImportError|ModuleNotFoundError/i, suggestions: [
    {label: 'Fix this error', action: '//fix this error'},
    {label: 'Install missing', action: '//install the missing dependency'},
    {label: 'Explain', action: '//explain this error and how to fix it'},
    {label: 'Undo', action: '//revert the last change'},
  ]},
  // Successful operations
  { match: /Successfully installed|added \d+ package|up to date/i, suggestions: [
    {label: 'Run the app', action: '//run the application'},
    {label: 'Run tests', action: '//run tests'},
    {label: 'What next?', action: '//what should I build next?'},
    {label: 'Show files', action: 'ls -la'},
  ]},
  { match: /\d+ passed.*0 failed|\d+ passed/i, suggestions: [
    {label: 'Commit', action: '//commit with a descriptive message'},
    {label: 'Add more tests', action: '//add more test coverage'},
    {label: 'Show diff', action: 'git diff'},
    {label: 'What next?', action: '//what else should I improve?'},
  ]},
  // Git status
  { match: /modified:|Untracked files|Changes not staged/i, suggestions: [
    {label: 'Commit all', action: '//commit all changes with a good message'},
    {label: 'Show diff', action: 'git diff'},
    {label: 'Stage all', action: 'git add -A && git status'},
    {label: 'Discard changes', action: '//discard uncommitted changes'},
  ]},
  // Agent completion
  { match: /━━━ Done/i, suggestions: [
    {label: 'Run tests', action: '//verify the changes by running tests'},
    {label: 'Commit', action: '//commit the changes'},
    {label: 'Review changes', action: 'git diff'},
    {label: 'What else?', action: '//what else should be done?'},
  ]},
  // Agent error
  { match: /━━━ Error|Agent error|error\(s\)/i, suggestions: [
    {label: 'Try again', action: '//try that again with a different approach'},
    {label: 'Explain error', action: '//what went wrong?'},
    {label: 'Undo', action: '//revert to the last checkpoint'},
    {label: 'Manual fix', action: '//show me the file so I can fix it manually'},
  ]},
  // Server running
  { match: /Running on|Listening on|localhost:\d+|0\.0\.0\.0:\d+/i, suggestions: [
    {label: 'Test endpoint', action: '//test the main endpoint with curl'},
    {label: 'Show logs', action: '//what do the server logs say?'},
    {label: 'Stop server', action: '\x03'},  // Ctrl+C
    {label: 'Add feature', action: '//add a new feature to this app'},
  ]},
  // Empty workspace
  { match: /total 0|total 8\n/i, suggestions: [
    {label: 'Start a project', action: '?:What kind of project?://set up a new project: ${input}'},
    {label: 'Clone a repo', action: '?:Paste a repo URL or describe what to clone://clone ${input}'},
    {label: 'Create app', action: '?:Describe the app you want to build://create a web application: ${input}'},
    {label: 'Help', action: '//what kinds of projects can you help me build?'},
  ]},
];

const _DEFAULT_SUGGESTIONS = [
  {label: "What's here?", action: '//list the files here and describe what this workspace contains'},
  {label: 'Create something', action: '?:What would you like to create?://create the following: ${input}'},
  {label: 'Install tools', action: '?:What language or tools do you need?://set up a development environment for ${input}'},
  {label: 'Help', action: '//what can you help me with in this workspace?'},
];

function _extractQuestionOptions(text) {
  // Extract bullet-point or numbered options from agent's question
  // Matches: "- Option text", "1. Option text", "• Option text"
  const options = [];
  const lines = text.split('\n');
  for (const line of lines) {
    const trimmed = line.trim();
    // Match "- text", "• text", "1. text", "a) text"
    const match = trimmed.match(/^[-•*]\s+(.+)$/) ||
                  trimmed.match(/^\d+[.)]\s+(.+)$/) ||
                  trimmed.match(/^[a-d][.)]\s+(.+)$/i);
    if (match && match[1] && match[1].length > 3 && match[1].length < 60) {
      options.push(match[1].trim());
    }
  }
  return options;
}

function _setIntentButtons(suggestions) {
  if (!currentCoder().dom.intentBar) return;
  const buttons = currentCoder().dom.intentBar.querySelectorAll('.coder-intent-btn');
  suggestions.forEach((s, i) => {
    if (buttons[i]) {
      buttons[i].textContent = s.label;
      buttons[i].dataset.action = s.action;
      buttons[i].style.display = '';
    }
  });
  // Hide unused buttons
  for (let i = suggestions.length; i < buttons.length; i++) {
    if (buttons[i]) buttons[i].style.display = 'none';
  }
}

function _updateIntentBar() {
  if (!currentCoder().dom.intentBar || !currentCoder().terminalId) return;

  const scrollback = Terminal.getScrollback(currentCoder().terminalId, 10);

  let suggestions = _DEFAULT_SUGGESTIONS;
  for (const pattern of _INTENT_PATTERNS) {
    if (pattern.match.test(scrollback)) {
      suggestions = pattern.suggestions;
      break;
    }
  }

  const buttons = currentCoder().dom.intentBar.querySelectorAll('.coder-intent-btn');
  suggestions.forEach((s, i) => {
    if (buttons[i]) {
      buttons[i].textContent = s.label;
      buttons[i].dataset.action = s.action;
    }
  });
}

// ---------------------------------------------------------------------------
// Workspace Switcher
// ---------------------------------------------------------------------------

async function _populateWorkspaceSelect(workspaces = null) {
  const select = document.getElementById('coder-workspace-select');
  if (!select) return;
  // Accept a pre-fetched list so callers in _onEnterCoderMode can
  // share one /api/coder/workspaces round-trip across the select +
  // metadata bind + status chips. Falls back to fetching when called
  // standalone (e.g. the 10s status poller, post-create refresh).
  if (workspaces === null) workspaces = await _fetchWorkspaces();
  const optionsHtml = workspaces.length === 0
    ? '<option value="">No workspaces</option>'
    : workspaces.map(w =>
      `<option value="${escapeHtml(w.id)}" ${w.id === currentCoder().workspaceId ? 'selected' : ''}>${escapeHtml(w.name)} (${w.status})</option>`
    ).join('');
  // Skip the DOM write when nothing changed — this runs on a 10s status
  // poll, and an innerHTML rebuild of an unchanged list is pure churn
  // (it also closes the dropdown if the activeElement guard ever misses).
  if (select.dataset.optionsSig !== optionsHtml) {
    select.innerHTML = optionsHtml;
    select.dataset.optionsSig = optionsHtml;
  }
}

// Monotonic counter bumped on every _switchWorkspace entry. Each in-flight
// switch captures its value on entry and re-checks across every ``await``
// so a later switch supersedes any older one mid-flight (rapid A→B→C
// clicks → only C's terminal + title + tree win, A and B exit early
// without clobbering newer state).
let _switchGeneration = 0;

async function _switchWorkspace(workspaceId) {
  if (workspaceId === currentCoder().workspaceId) return;

  // Unsaved editor buffers live only in memory and are dropped on switch —
  // snapshot the active one first so the dirty check sees the latest, then
  // guard. Bailing restores the dropdown to the current workspace.
  _snapshotActiveEditorBuffer();
  if (currentCoder().editorFiles.some((f) => f.dirty) &&
      !confirm('You have unsaved changes in this workspace. Switch and discard them?')) {
    const sel = document.getElementById('coder-workspace-select');
    if (sel && currentCoder().workspaceId) sel.value = currentCoder().workspaceId;
    return;
  }

  const myGen = ++_switchGeneration;
  const superseded = () => myGen !== _switchGeneration;

  // Capture the OLD workspace's state synchronously — currentCoder().conversation is
  // about to be cleared and we still need to send its messages to the
  // server. Skipped when there's no prior workspace (first switch).
  const oldId = currentCoder().workspaceId;
  const oldHistory = (oldId && currentCoder().conversation) ? currentCoder().conversation.getHistory() : null;

  // Kick off save (PUT) and xterm CDN load in parallel — they're
  // independent of each other and were the two big serial awaits
  // dominating switch latency. Save is awaited later (required for
  // the A→B→A correctness contract: if we re-open A right after B,
  // the GET must see A's saved state, not the pre-save state).
  //
  // Don't save when the snapshot is empty. Without this guard a
  // rapid A→B→C path would race like so: B clears currentCoder().conversation
  // before its save settles; C arrives, reads currentCoder().workspaceId=B
  // and an empty currentCoder().conversation, fires a save against B with an
  // empty body — destructively overwriting B's prior server state.
  // Empty saves carry no information anyway, so skipping them is
  // both safe and faster.
  const hasMeaningfulHistory = !!(oldId && oldHistory && oldHistory.length > 0);
  const savePromise = hasMeaningfulHistory
    ? fetch(`/api/coder/conversation/${encodeURIComponent(oldId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: oldHistory }),
      }).catch(() => null)
    : Promise.resolve(null);
  const xtermPromise = Terminal.load().catch(() => null);

  // Destroy current terminal (sync) — safe to do before save settles
  // since the WS is tied to the OLD workspace and we're leaving it.
  if (currentCoder().terminalId) {
    Terminal.destroy(currentCoder().terminalId);
    currentCoder().terminalId = null;
  }

  // Destroy current editor and hide split
  if (currentCoder().activeEditorId) {
    Editor.destroy(currentCoder().activeEditorId);
    currentCoder().activeEditorId = null;
    currentCoder().activeFilePath = '';
    if (currentCoder().dom.editorPane) currentCoder().dom.editorPane.innerHTML = '';
  }
  currentCoder().editorFiles = [];
  currentCoder().activeEditorFile = null;
  _hideEditorSplit();

  // Clear chat history (different workspace = different context).
  // Also clear the visible conversation — _loadConversation now guards
  // against clobbering an in-flight turn by skipping when the pane is
  // non-empty, so workspace-switch paths MUST clear explicitly to give
  // the fresh load a blank canvas to write into. Without this, the
  // previous workspace's messages would linger under the new one.
  currentCoder().chatHistory = [];
  currentCoder().conversation?.clear();

  // Drop any mission state from the previous workspace — fresh context.
  currentCoder().missionPanel?.clear();
  _resetPreviewState();
  _setWorkbenchTab('terminal');

  // Search results belong to the workspace we're leaving — close the
  // pane so a stale hit list can't be clicked into the new workspace.
  if (isCoderSearchOpen()) closeCoderSearch();
  // Drop any multi-selection from the previous workspace's tree.
  if (_selectedPaths.size) _clearSelection();

  currentCoder().workspaceId = workspaceId;
  _persistActiveWorkspaceId(workspaceId);
  // Tell the conversation so tool-result renderers (browser_screenshot
  // inline embed, future workspace-scoped artifacts) can build correct
  // workspace-relative URLs without sniffing the route.
  currentCoder().conversation?.setWorkspaceId(workspaceId);

  // Instant feedback: update title from the dropdown's selected option
  // BEFORE any await fires, so the user sees the new workspace name
  // immediately rather than after the terminal handshake settles.
  if (currentCoder().dom.filesTitle) {
    const select = document.getElementById('coder-workspace-select');
    const name = select?.selectedOptions[0]?.textContent || 'Files';
    currentCoder().dom.filesTitle.textContent = name.split(' (')[0]; // Just the name, not the status
  }

  // Wait for the save to finish so A→B→A correctness holds (a re-open
  // of the old workspace later in this function must see the saved
  // state, not pre-save). Concurrency-superseded switches exit here.
  await savePromise;
  if (superseded()) return;

  // Kick off every fire-and-forget data load BEFORE the terminal
  // create await — file tree, conversation history, checkpoints, etc.
  // race the WS handshake instead of sitting behind it. The visible
  // chrome populates while xterm is still negotiating its socket.
  if (workspaceId) {
    _populateFileTree(workspaceId);
    _loadConversation(workspaceId);
    // Show index-build progress if a build is already running for this
    // workspace (e.g. kicked off on coder-mode entry). Self-stops if idle.
    _startIndexProgressPoll(workspaceId);
    // Reattach to an in-flight coder run if one is still going on
    // the server (mobile screen wake, tab restore, laptop unlid).
    // No-op when the workspace is idle.
    void _attemptCoderReconnect(workspaceId);
    // Refresh checkpoint count so the status-bar tile reflects the
    // new workspace. List remains collapsed unless user expanded it.
    _loadCheckpoints();
    // Services panel mirrors the active workspace — re-load + restart
    // the 8s poll so the user-controlled toggles bind to the right
    // services without a coder-mode round-trip.
    _loadServices();
    _startServicesPolling();
    // Preview trust banner re-renders against the new workspace's
    // trust flag (per-workspace localStorage). Without this, switching
    // from a trusted workspace to an untrusted one would leave the
    // banner hidden and the user wondering why HMR is broken.
    _refreshPreviewTrustUi();
    _refreshPowerTile();
    _refreshActiveSafeguards();
    // Pull the persisted planning_mode for this workspace so the
    // badge in the composer matches what the server will use on the
    // next turn. Best-effort — defaults to "default" on fetch error.
    void _refreshPlanningModeBadge(workspaceId);
    // Thinking toggle is per-workspace (persisted in localStorage)
    // and depends on the current model's capability — refresh both
    // visibility + state on every workspace switch.
    _refreshCoderThinkingToggle();
  }
  _startGitPolling();

  // Connect new terminal — xterm load was started at the top, almost
  // always already settled by the time we await here.
  if (currentCoder().dom.terminalPane && workspaceId) {
    currentCoder().dom.terminalPane.innerHTML = '';
    await xtermPromise;
    if (superseded()) return;
    const nextTerminalId = await Terminal.create(currentCoder().dom.terminalPane, workspaceId);
    if (superseded()) {
      // A newer switch ran while Terminal.create was awaiting. Clean
      // up our just-built terminal so we don't leak it into the next
      // workspace's pane.
      Terminal.destroy(nextTerminalId);
      return;
    }
    currentCoder().terminalId = nextTerminalId;
    Terminal.focus(currentCoder().terminalId);
  }

  _updateStatus('idle');
  _updateIntentBar();
  _refreshPowerTile();

  // Notify subscribers (inspector, future surfaces) so they can
  // re-bind to the new workspace. currentCoder().workspaceId may be empty
  // here if the user just deleted the last workspace.
  document.dispatchEvent(new CustomEvent('coder-workspace-changed', {
    detail: { workspaceId: workspaceId || '' },
  }));
}

export async function openWorkspaceById(workspaceId) {
  if (!workspaceId) return;
  if (currentCoder().workspaceId === workspaceId) {
    await _populateWorkspaceSelect();
    if (workspaceId) _populateFileTree(workspaceId);
    return;
  }
  await _switchWorkspace(workspaceId);
  await _populateWorkspaceSelect();
}

// ---------------------------------------------------------------------------
// File Upload / Download / Workspace Export
// ---------------------------------------------------------------------------

// Recursively walk a DataTransferItem entry (file or directory) and
// collect {file, path} pairs. Path preserves the dropped folder's
// structure so it lands at /workspace/<folder>/... via Docker's tar
// extraction, not flattened at the root.
async function _walkDropEntry(entry, prefix, out) {
  if (entry.isFile) {
    const file = await new Promise((ok, fail) => entry.file(ok, fail));
    out.push({ file, path: prefix + entry.name });
    return;
  }
  if (entry.isDirectory) {
    const reader = entry.createReader();
    // Chrome's readEntries returns batches of 100; loop until empty.
    while (true) {
      const batch = await new Promise((r) => reader.readEntries(r));
      if (!batch || batch.length === 0) break;
      for (const child of batch) {
        await _walkDropEntry(child, prefix + entry.name + '/', out);
      }
    }
  }
}

async function _uploadFiles(files) {
  const wid = currentCoder().workspaceId;
  if (!wid) { showToast('No workspace selected', 'warning'); return; }
  if (!files.length) return;
  const form = new FormData();
  form.append('dest_path', '/workspace');
  for (const { file, path } of files) {
    form.append('files', file, path);
  }
  showToast(`Uploading ${files.length} file${files.length === 1 ? '' : 's'}...`, 'info', 3000);
  try {
    const resp = await fetch(`/api/coder/files/${encodeURIComponent(wid)}/upload`, {
      method: 'POST',
      body: form,
    });
    const data = await resp.json();
    if (!resp.ok) {
      showToast(extractErrorMessage(data, 'Upload failed'), 'error', 5000);
      return;
    }
    const skipped = data.skipped?.length ? ` (${data.skipped.length} skipped)` : '';
    showToast(`Uploaded ${data.uploaded} file${data.uploaded === 1 ? '' : 's'}${skipped}`, 'success');
    _populateFileTree(wid);
  } catch (err) {
    showToast('Upload failed: ' + err.message, 'error', 5000);
  }
}

function _initUploadAndDrop() {
  // Click-to-upload via a hidden <input>. Dual-purpose: single files
  // via the normal picker, entire folders via webkitdirectory. Clicking
  // the button with modifier keys could switch modes later; for now we
  // just offer files.
  const hiddenInput = document.createElement('input');
  hiddenInput.type = 'file';
  hiddenInput.multiple = true;
  hiddenInput.style.display = 'none';
  hiddenInput.addEventListener('change', async () => {
    const fileList = Array.from(hiddenInput.files || []);
    const items = fileList.map(f => ({
      file: f,
      // webkitRelativePath is populated when user picks a directory;
      // fall back to plain name for individual file picks.
      path: f.webkitRelativePath || f.name,
    }));
    if (items.length) await _uploadFiles(items);
    hiddenInput.value = '';
  });
  document.body.appendChild(hiddenInput);
  document.getElementById('coder-upload-btn')?.addEventListener('click', () => hiddenInput.click());

  // Drag-and-drop onto the file tree. Uses a depth counter to avoid
  // the dragenter/dragleave bouncing you get when the pointer moves
  // between child entries.
  const tree = currentCoder().dom.fileTree;
  if (!tree) return;
  let dragDepth = 0;
  const clearActive = () => {
    dragDepth = 0;
    tree.classList.remove('coder-file-tree--drop-active');
  };
  tree.addEventListener('dragenter', (e) => {
    if (!currentCoder().workspaceId || !e.dataTransfer?.types?.includes('Files')) return;
    e.preventDefault();
    dragDepth += 1;
    tree.classList.add('coder-file-tree--drop-active');
  });
  tree.addEventListener('dragover', (e) => {
    // Internal move onto empty tree area → move to workspace root.
    if (e.dataTransfer?.types?.includes(_DRAG_TYPE)) {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      return;
    }
    if (!currentCoder().workspaceId || !e.dataTransfer?.types?.includes('Files')) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  });
  tree.addEventListener('dragleave', () => {
    dragDepth -= 1;
    if (dragDepth <= 0) clearActive();
  });
  tree.addEventListener('drop', async (e) => {
    // Internal move to the workspace root. Folder rows handle their own
    // drops (with stopPropagation); anything reaching here was dropped on
    // empty tree space, meaning "move to /workspace".
    if (e.dataTransfer?.types?.includes(_DRAG_TYPE)) {
      e.preventDefault();
      clearActive();
      if (!currentCoder().workspaceId) return;
      const payload = _readInternalDrag(e.dataTransfer);
      if (payload?.multi) {
        await _moveMany(currentCoder().workspaceId, payload.multi, '/workspace');
      } else if (payload) {
        await _moveEntry(currentCoder().workspaceId, payload.path, '/workspace', { isDir: payload.isDir });
      }
      return;
    }
    e.preventDefault();
    clearActive();
    if (!currentCoder().workspaceId) return;
    const out = [];
    const items = e.dataTransfer?.items;
    if (items && items.length) {
      for (const it of items) {
        if (it.kind !== 'file') continue;
        const entry = it.webkitGetAsEntry?.();
        if (entry) {
          await _walkDropEntry(entry, '', out);
        } else {
          const file = it.getAsFile();
          if (file) out.push({ file, path: file.name });
        }
      }
    } else {
      for (const file of (e.dataTransfer?.files || [])) {
        out.push({ file, path: file.name });
      }
    }
    if (out.length) await _uploadFiles(out);
  });

  // Block drops outside the tree from navigating away (default browser
  // behavior is to open the file). Scoped to the files panel, not
  // window-wide, to avoid stepping on other drop targets.
  const panel = document.getElementById('coder-files-view');
  panel?.addEventListener('dragover', (e) => {
    if (e.dataTransfer?.types?.includes('Files')) e.preventDefault();
  });
  panel?.addEventListener('drop', (e) => {
    if (e.dataTransfer?.types?.includes('Files')) e.preventDefault();
  });
}

function _downloadWorkspace() {
  if (!currentCoder().workspaceId) { showToast('No workspace selected', 'warning'); return; }
  const url = `/api/coder/workspaces/${encodeURIComponent(currentCoder().workspaceId)}/export`;
  // Trigger a browser download. Streaming happens inside the browser —
  // the tar.gz is generated lazily on the server, so large workspaces
  // don't buffer in memory.
  const a = document.createElement('a');
  a.href = url;
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
  showToast('Preparing archive… (node_modules / .venv excluded by default)', 'info', 4000);
}

// ---------------------------------------------------------------------------
// Expansion Prompt — ask user for details before sending to agent
// ---------------------------------------------------------------------------

/**
 * Parse a "?:prompt://template ${input}" action, show the prompt in the
 * terminal, collect the user's typed answer, then compose the final agent
 * request from the template and run it.
 *
 * Format: "?:Question text here://agent instruction with ${input}"
 */
function _promptAndRun(action) {
  if (!currentCoder().terminalId) return;

  // Parse "?:prompt://template"
  const parts = action.slice(2).split('://');
  if (parts.length < 2) return;
  const promptText = parts[0];
  const template = parts.slice(1).join('://'); // rejoin in case template has ://

  // Show the prompt in the terminal
  Terminal.write(currentCoder().terminalId,
    `\r\n\x1b[1;36m${promptText}\x1b[0m\r\n\x1b[36m▸ \x1b[0m`);

  // Temporarily capture terminal input until Enter is pressed
  let inputBuf = '';
  const tid = currentCoder().terminalId;
  const term = Terminal.getTerminalInstance(tid);
  if (!term) return;

  // Block normal input handling
  Terminal.setAgentActive(tid, true);

  currentCoder().activePromptDisposable = term.onData((data) => {
    if (data === '\r' || data === '\n') {
      // Enter — finalize
      currentCoder().activePromptDisposable.dispose();
      currentCoder().activePromptDisposable = null;
      Terminal.write(tid, '\r\n');
      Terminal.setAgentActive(tid, false);

      const userInput = inputBuf.trim();
      if (!userInput) {
        _updateIntentBar();
        return;
      }
      const finalRequest = template.replace(/\$\{input\}/g, userInput);
      if (finalRequest.startsWith('//')) {
        _runAgentInTerminal(finalRequest.slice(2));
      } else {
        _runAgentInTerminal(finalRequest);
      }
    } else if (data === '\x7f' || data === '\b') {
      if (inputBuf.length > 0) {
        inputBuf = inputBuf.slice(0, -1);
        Terminal.write(tid, '\b \b');
      }
    } else if (data === '\x03') {
      currentCoder().activePromptDisposable.dispose();
      currentCoder().activePromptDisposable = null;
      Terminal.write(tid, '\r\n\x1b[33m[Cancelled]\x1b[0m\r\n');
      Terminal.setAgentActive(tid, false);
      _updateIntentBar();
    } else if (data >= ' ') {
      inputBuf += data;
      Terminal.write(tid, data);
    }
  });

  _updateStatus('idle', 'waiting for your input');
  _setIntentButtons([{label: 'Cancel (Esc)', action: '__cancel'}]);
}

// ---------------------------------------------------------------------------
// Conversation Input — sends from the conversation input bar
// ---------------------------------------------------------------------------

/**
 * Upload dropped / pasted files into the appropriate pipeline, render
 * chips in the composer, and append to currentCoder().pendingAttachments.
 *
 * Optimistic render: a chip with a progress bar appears IMMEDIATELY
 * when the drop/paste lands, then the upload runs in the background
 * and updates the bar. Uploads run in parallel across the dropped
 * batch so a 20-file drop doesn't serialise; each chip updates
 * independently as its upload completes.
 *
 * Failure handling: if a single upload fails, its chip briefly shows
 * the failed state (red border) then is removed. Other files in the
 * same batch aren't affected.
 */
async function _ingestAttachments(files) {
  if (!files?.length) return;
  const mod = await import('./coder-attachments.js');
  const row = document.getElementById('coder-attachments-row');
  if (!row) return;

  // Helper the remove-X button uses. Shared across pending + ready
  // states so removing a mid-upload chip works too (we just drop the
  // descriptor; the XHR keeps running but its result is ignored —
  // acceptable for MVP, an AbortController upgrade could cancel mid-
  // flight if needed).
  const makeRemoveHandler = (chipEl, descRef) => (target) => {
    const idx = currentCoder().pendingAttachments.findIndex(a => a.id === descRef.current.id);
    if (idx >= 0) currentCoder().pendingAttachments.splice(idx, 1);
    chipEl.remove();
    if (!currentCoder().pendingAttachments.length) row.classList.add('hidden');
  };

  // Render optimistic chips FIRST so the UI is responsive before any
  // bytes move. Collect per-file upload promises and await them all
  // in parallel below.
  const tasks = await Promise.all(files.map(async (f) => {
    const pending = await mod.buildPendingDescriptor(f);
    currentCoder().pendingAttachments.push(pending);
    // `current` indirection so the remove handler reads the LATEST
    // descriptor id after the pending-→-real swap (ids change at
    // swap time).
    const descRef = { current: pending };
    const chipEl = mod.renderChip(pending);
    const removeBtn = chipEl.querySelector('.coder-attachment-remove');
    if (removeBtn) {
      removeBtn.onclick = (e) => {
        e.stopPropagation();
        makeRemoveHandler(chipEl, descRef)(descRef.current);
      };
    }
    row.appendChild(chipEl);
    row.classList.remove('hidden');
    return { file: f, pending, descRef, chipEl };
  }));

  // Fire uploads in parallel. Each completes independently and
  // updates its own chip. Using Promise.all for cleanup semantics,
  // but allSettled would also work — failures don't cascade.
  await Promise.all(tasks.map(async ({ file, pending, descRef, chipEl }) => {
    try {
      const att = await mod.ingestFile(
        file, currentCoder().workspaceId, currentCoder().workspaceId,
        (pct) => mod.updateChipProgress(chipEl, pct),
      );
      if (!att) {
        // Upload failed — flash error state, then remove after a
        // moment so the user sees WHY.
        mod.markChipFailed(chipEl);
        const idx = currentCoder().pendingAttachments.indexOf(pending);
        if (idx >= 0) currentCoder().pendingAttachments.splice(idx, 1);
        setTimeout(() => {
          chipEl.remove();
          if (!currentCoder().pendingAttachments.length) row.classList.add('hidden');
        }, 1500);
        return;
      }
      // Replace pending descriptor with the real one in both the
      // array (for the send path) and the descRef (for the remove
      // handler).
      const idx = currentCoder().pendingAttachments.indexOf(pending);
      if (idx >= 0) currentCoder().pendingAttachments[idx] = att;
      descRef.current = att;
      chipEl.dataset.attachmentId = att.id;
      mod.updateChipProgress(chipEl, null);  // mark complete
    } catch (err) {
      console.warn('[coder-attachments] ingest failed', err);
      mod.markChipFailed(chipEl);
    }
  }));
}


/** Drain the pending attachments list + collapse the chip row. */
function _drainAttachments() {
  const list = currentCoder().pendingAttachments.splice(0, currentCoder().pendingAttachments.length);
  const row = document.getElementById('coder-attachments-row');
  if (row) {
    row.innerHTML = '';
    row.classList.add('hidden');
  }
  return list;
}


function _sendConversationMessage(opts = {}) {
  // Cooperative-mode router. ``opts.mode`` is one of:
  //   "auto"      — context-dependent. Idle → new turn. Running →
  //                 queue interjection (end-of-turn drain).
  //   "steer"     — explicit steer interjection (next-iteration
  //                 drain). Ignored when idle (sends as new turn
  //                 instead, since there's no run to steer into).
  //   "interrupt" — explicit cancel + send-as-new-turn. The
  //                 legacy "stop and replace" muscle memory.
  const mode = String(opts.mode || 'auto');

  const streamActive = currentCoder().coderStream?.isActive();
  const terminalAgentActive = !!(currentCoder().activePromptDisposable || currentCoder().terminalAgentAbort);

  // Explicit interrupt → cancel the current run + fall through to send.
  // This mirrors the pre-cooperative behavior the user already knows.
  if (mode === 'interrupt' && streamActive) {
    void _cancelActiveRunOnServer();
    currentCoder().coderStream.abort();
    currentCoder().conversation?.addError('Cancelled by user');
    _updateStatus('idle', 'cancelled');
    // Fall through to send the new content as a fresh turn.
  } else if (mode === 'interrupt' && terminalAgentActive) {
    // Terminal agent path doesn't have an inbox; interrupt is the
    // only available semantics there.
    if (currentCoder().activePromptDisposable) {
      currentCoder().activePromptDisposable.dispose();
      currentCoder().activePromptDisposable = null;
      if (currentCoder().terminalId) {
        Terminal.write(currentCoder().terminalId, '\r\n\x1b[33m[Cancelled]\x1b[0m\r\n');
        Terminal.setAgentActive(currentCoder().terminalId, false);
      }
    } else if (currentCoder().terminalAgentAbort) {
      try { currentCoder().terminalAgentAbort.abort(); } catch { /* noop */ }
      currentCoder().terminalAgentAbort = null;
    }
    _updateStatus('idle', 'cancelled');
    return;
  } else if ((mode === 'auto' || mode === 'steer' || mode === 'queue') && streamActive) {
    // Cooperative interjection. Send the typed content + drained
    // attachments to the inbox endpoint with the resolved drain mode.
    //
    // ``auto`` maps to ``steer`` (2026-05-31). The previous default
    // mapped to ``queue`` (end-of-turn drain → new chained turn),
    // which caused intent drift on long turns: user types a redirect
    // mid-turn, model keeps grinding through its current plan for
    // many more iterations, and by the time the queued message lands
    // it no longer matches the situation. ``steer`` drains at the
    // next iteration boundary (between tool-call batches), which is
    // the natural "next convenient seam" — the model finishes the
    // tool it's currently emitting, sees the steer message at the
    // top of the next iteration, and re-routes. Explicit ``queue``
    // is still available for callers that want the cleaner handoff
    // (the badge UX preserves the queued → delivered transition for
    // both).
    const drainMode = mode === 'queue' ? 'queue' : 'steer';
    void _sendInterjection(drainMode);
    return;
  } else if (mode === 'auto' && terminalAgentActive) {
    // Terminal agent path with no inbox — auto becomes cancel+send,
    // preserving the prior "send while running cancels" muscle
    // memory for that surface.
    if (currentCoder().activePromptDisposable) {
      currentCoder().activePromptDisposable.dispose();
      currentCoder().activePromptDisposable = null;
    } else if (currentCoder().terminalAgentAbort) {
      try { currentCoder().terminalAgentAbort.abort(); } catch { /* noop */ }
      currentCoder().terminalAgentAbort = null;
    }
    _updateStatus('idle', 'cancelled');
    // Fall through to send.
  }

  const input = document.getElementById('coder-input');
  if (!input) return;
  const text = input.value.trim();
  // Allow a pure-attachments send (no text, just a dropped image/file).
  // Attachments are required to have SOMETHING going out; empty text
  // + no attachments is a no-op.
  if (!currentCoder().workspaceId) return;
  if (!text && currentCoder().pendingAttachments.length === 0) return;

  input.value = '';
  input.style.height = 'auto';
  // Hide onboarding empty state
  document.getElementById('coder-conv-empty')?.remove();

  // Slash commands intercept before we hit the backend. Kept to
  // single-word commands for discoverability — anything else falls
  // through to a normal agent request so users can still talk about
  // slashes naturally. Attachments don't apply to slash commands —
  // bail before draining so a /help typed with an attached image
  // preserves the attachment for the user's next "real" send.
  if (text && _handleSlashCommand(text)) return;

  // Drain attachments into a payload for this turn.
  const attachments = _drainAttachments();

  // Use conversation path if available, fall back to terminal
  const layout = document.getElementById('coder-layout');
  if (currentCoder().conversation && layout && !layout.classList.contains('classic-mode')) {
    _runAgentInConversation(text, attachments);
  } else {
    _runAgentInTerminal(text);
  }
}

/**
 * Handle coder-mode slash commands. Returns true iff the input was
 * consumed (caller should NOT fall through to the normal send path).
 *
 * Supported:
 *   /clear  — wipe this workspace's conversation history + server-side
 *             plan/tasks state. Does not touch files in the container.
 *   /compact — summarize older conversation messages and keep recent context.
 *   /queue <task> — queue a headless background mission (jobs spine);
 *             notification on completion, live-watch via active-run reattach.
 *   /help   — render a cheat-sheet of coder-mode features + workspace
 *             tooling (the stuff users don't know about because we
 *             never told them).
 */
function _handleSlashCommand(text) {
  const trimmed = text.trim();
  // Single-word commands only for v1. "/clear foo" is still /clear.
  const cmd = trimmed.split(/\s+/, 1)[0];

  if (cmd === '/clear') {
    _clearCoderConversation().catch(err => {
      console.warn('clear_conversation failed', err);
      currentCoder().conversation?.addError('Failed to clear conversation. Try refreshing the page.');
    });
    return true;
  }

  // /claude <task> — run the task with Claude Code inside this workspace's
  // sandbox (uses your connected token). Opens the Agents panel and kicks the
  // run; results render there. With no task, just opens the panel to connect.
  if (cmd === '/claude') {
    const task = trimmed.slice(cmd.length).trim();
    const modal = document.getElementById('coder-agents-modal');
    if (modal) modal.classList.remove('hidden');
    _refreshClaudeAgentStatus();
    _initComposer();
    if (task) {
      currentCoder().conversation?.addUserMessage(`/claude ${task}`);
      const taskEl = document.getElementById('cca-task');
      if (taskEl) taskEl.value = task;
      _runWithClaude();
    }
    return true;
  }

  if (cmd === '/compact') {
    _compactCoderConversation().catch(err => {
      console.warn('compact_conversation failed', err);
      currentCoder().conversation?.addError('Failed to compact conversation. Try refreshing the page.');
    });
    return true;
  }

  // /queue <task> — run the task as a headless background mission via the
  // jobs spine. The run uses THIS workspace + the model currently selected
  // in the model picker (chosen by the user — never auto-picked), executes
  // under the workspace's normal permission policy, and lands a
  // notification (coder.run.complete / coder.run.failed) when done. Watch
  // live any time by reopening the workspace (active-run reattach).
  if (cmd === '/queue') {
    const task = trimmed.slice(cmd.length).trim();
    if (!task) {
      currentCoder().conversation?.addError('Usage: /queue <task> — queues a background mission for this workspace.');
      return true;
    }
    _queueBackgroundRun(task).catch(err => {
      console.warn('queue_background_run failed', err);
      currentCoder().conversation?.addError(err?.message || 'Failed to queue background mission.');
    });
    return true;
  }

  if (cmd === '/help' || cmd === '/?') {
    _showCoderHelp();
    return true;
  }

  if (cmd === '/powers') {
    _showCoderPowers().catch(err => {
      console.warn('powers_list failed', err);
      currentCoder().conversation?.addError('Failed to load Powers.');
    });
    return true;
  }

  if (cmd === '/power') {
    _handlePowerSlash(trimmed).catch(err => {
      console.warn('power command failed', err);
      currentCoder().conversation?.addError(err?.message || 'Power command failed.');
    });
    return true;
  }

  return false;
}

/**
 * Render the coder-mode cheat-sheet inline in the conversation pane.
 *
 * Teaches features that currently have zero discoverability: slash
 * commands, the ``//`` terminal prefix, xvfb-run for GUI code,
 * pre-installed tooling. Doesn't hit the server — purely local.
 *
 * The same content feeds the first-time empty-state so the user
 * sees it before ever asking anything; ``/help`` lets them recall
 * it after the first send has scrolled it off.
 */
/**
 * Queue a headless background mission for the active workspace.
 *
 * Wiring: POST /api/coder/workspaces/{id}/background-runs → jobs spine
 * (job_type coder_background_run) → the normal coder turn stack with the
 * job runner as the client. Model is REQUIRED and comes from the user's
 * current picker selection — no silent default. The queued/running state
 * is visible via the jobs surface (GET /api/jobs/?type=coder_background_run)
 * and the finished run lands a notification; reopening the workspace
 * while it runs reattaches live via the existing active-run path.
 */
async function _queueBackgroundRun(task) {
  if (!currentCoder().workspaceId) {
    currentCoder().conversation?.addError('No active workspace — open one before queuing a mission.');
    return;
  }
  const model = app.state.currentModel || '';
  if (!model) {
    currentCoder().conversation?.addError('Pick a model first — background missions run with your selected model.');
    return;
  }
  const resp = await fetch(
    `/api/coder/workspaces/${encodeURIComponent(currentCoder().workspaceId)}/background-runs`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: task, model }),
    },
  );
  if (!resp.ok) {
    const err = await resp.text().catch(() => resp.statusText);
    throw new Error(err || `Queue failed (${resp.status})`);
  }
  const data = await resp.json();
  // Event line (not a chat message — must not enter getMessagesForLLM and
  // confuse the model's view of the conversation).
  _appendCoderEventHtml(
    `<div class="coder-event-title">Queued background mission</div>` +
    `<div class="coder-event-body">` +
    `<code>${escapeHtml(task.length > 80 ? task.slice(0, 79) + '…' : task)}</code>` +
    `<span class="coder-event-meta">${escapeHtml(model)} · job ${escapeHtml(data.job_id || '?')} · ` +
    `notification on completion — keep this workspace open to watch live.</span>` +
    `</div>`,
  );
  showToast('Mission queued — notification on completion', 'success');
}

function _showCoderHelp() {
  if (!currentCoder().conversation) return;
  document.getElementById('coder-conv-empty')?.remove();
  const el = document.createElement('div');
  el.className = 'coder-msg coder-msg-assistant coder-msg-help';
  el.innerHTML = _CODER_HELP_HTML;
  const host = document.getElementById('coder-conv-messages');
  if (host) {
    host.appendChild(el);
    currentCoder().conversation._scrollToBottom?.();
  }
}

function _appendCoderHelpHtml(html) {
  if (!currentCoder().conversation) return;
  document.getElementById('coder-conv-empty')?.remove();
  const el = document.createElement('div');
  el.className = 'coder-msg coder-msg-assistant coder-msg-help';
  el.innerHTML = html;
  const host = document.getElementById('coder-conv-messages');
  if (host) {
    host.appendChild(el);
    currentCoder().conversation._scrollToBottom?.();
  }
}

function _appendCoderEventHtml(html) {
  if (!currentCoder().conversation) return;
  document.getElementById('coder-conv-empty')?.remove();
  const el = document.createElement('div');
  el.className = 'coder-msg coder-msg-event';
  el.innerHTML = html;
  const host = document.getElementById('coder-conv-messages');
  if (host) {
    host.appendChild(el);
    currentCoder().conversation._scrollToBottom?.();
  }
}

async function _showCoderPowers() {
  const mod = await import('./powers.js');
  const data = await mod.fetchPowers(currentCoder().workspaceId || '');
  _appendCoderHelpHtml(mod.powersHelpHtml(data, { workspaceId: currentCoder().workspaceId || '' }));
  _refreshPowerTile();
}

async function _handlePowerSlash(text) {
  const mod = await import('./powers.js');
  const arg = text.replace(/^\/power\b/i, '').trim();
  if (!arg) {
    const data = await mod.fetchPowers(currentCoder().workspaceId || '');
    _appendCoderHelpHtml(mod.powersHelpHtml(data, { workspaceId: currentCoder().workspaceId || '' }));
    return;
  }
  if (arg === 'off') {
    await mod.clearActivePower(currentCoder().workspaceId || '');
    showToast('Power cleared', 'success');
    _appendCoderHelpHtml(mod.activePowerWhyHtml({ power: null }));
    _refreshPowerTile();
    return;
  }
  if (arg === 'why') {
    const resp = await fetch(`/api/powers/active${currentCoder().workspaceId ? `?workspace_id=${encodeURIComponent(currentCoder().workspaceId)}` : ''}`);
    if (!resp.ok) throw new Error('Failed to load active Power');
    const payload = await resp.json();
    _appendCoderHelpHtml(mod.activePowerWhyHtml(payload));
    return;
  }

  const data = await mod.fetchPowers(currentCoder().workspaceId || '');
  const needle = arg.toLowerCase();
  const exact = data.powers.find(p => p.id.toLowerCase() === needle || p.display_name.toLowerCase() === needle);
  const fuzzy = exact || data.powers.find(p => p.id.toLowerCase().includes(needle) || p.display_name.toLowerCase().includes(needle));
  if (!fuzzy) {
    throw new Error(`No Power matched "${arg}"`);
  }
  await mod.activatePower(fuzzy.id, currentCoder().workspaceId || '');
  showToast(`Power activated: ${fuzzy.display_name}`, 'success');
  try {
    const resp = await fetch(`/api/powers/active${currentCoder().workspaceId ? `?workspace_id=${encodeURIComponent(currentCoder().workspaceId)}` : ''}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const payload = await resp.json();
    _appendCoderHelpHtml(mod.activePowerWhyHtml(payload));
    _refreshPowerTile();
  } catch (err) {
    console.debug('[coder] active-power fetch failed', err);
  }
}

/**
 * Static HTML for ``/help`` and the empty-state placeholder. Kept
 * as a module constant so both surfaces render identical content.
 * Content is curated — only mentions features the user can actually
 * do from inside the conversation pane, not internal machinery.
 */
const _CODER_HELP_HTML = `
<div class="coder-help-title">Coder cheat-sheet</div>

<div class="coder-help-section">
  <div class="coder-help-heading">Slash commands</div>
  <ul class="coder-help-list">
    <li><code>/clear</code> — wipe conversation + plan state (files in the container are untouched).</li>
    <li><code>/compact</code> — summarize older conversation turns and keep the newest context live.</li>
    <li><code>/queue &lt;task&gt;</code> — run the task as a background mission with your selected model. You get a notification when it finishes; reopen the workspace anytime to watch live. Runs under this workspace's permission policy (permission prompts auto-deny after 60s when nobody's watching).</li>
    <li><code>/claude &lt;task&gt;</code> — run the task with Claude Code in this workspace's sandbox (uses your connected token; opens the Agents panel). <code>/claude</code> alone opens it to connect.</li>
    <li><code>/help</code> — show this cheat-sheet.</li>
    <li><code>/powers</code> — list installed Powers.</li>
    <li><code>/power &lt;id&gt;</code> — activate a Power for this workspace.</li>
  </ul>
</div>

<div class="coder-help-section">
  <div class="coder-help-heading">Terminal shortcut</div>
  <ul class="coder-help-list">
    <li>Type <code>// &lt;what you want done&gt;</code> at the terminal prompt to send an agent request that includes the last 20 lines of terminal scrollback as context. Handy for "fix that error above" without copy-pasting.</li>
  </ul>
</div>

<div class="coder-help-section">
  <div class="coder-help-heading">Headless GUI (turtle, tkinter, pygame, matplotlib)</div>
  <ul class="coder-help-list">
    <li>The workspace container has no real display. Run GUI code with <code>xvfb-run python3 app.py</code>.</li>
    <li>For matplotlib, <code>matplotlib.use('Agg')</code> + saving to PNG is cheaper than xvfb.</li>
  </ul>
</div>

<div class="coder-help-section">
  <div class="coder-help-heading">Pre-installed tools (don't apt-get reflexively)</div>
  <ul class="coder-help-list">
    <li><b>Languages:</b> python3 (+ pip, venv, tkinter, dev headers), node + npm, go, build-essential.</li>
    <li><b>CLIs:</b> git, git-lfs, curl, wget, jq, httpie, tree, ripgrep (<code>rg</code>), fd, unzip.</li>
    <li><b>Databases:</b> sqlite3, psql (postgresql-client), redis-cli.</li>
    <li><b>Python dev:</b> pytest, ruff, black, mypy, requests, httpx, flask, fastapi, uvicorn.</li>
    <li><b>Node dev:</b> typescript, ts-node, eslint, prettier (all global).</li>
    <li><b>Images:</b> imagemagick.</li>
  </ul>
</div>

<div class="coder-help-section">
  <div class="coder-help-heading">Powers — bias the agent at safe checkpoints</div>
  <ul class="coder-help-list">
    <li>Powers are capability packs the agent considers at planning / verification / workflow checkpoints — e.g. <code>multi-agent-review</code> for shipping-quality subsystem audits.</li>
    <li><code>/powers</code> — list installed Powers. <code>/power &lt;id&gt;</code> — activate one for this workspace. <code>/power off</code> — clear. <code>/power why</code> — explain the active one.</li>
    <li>You can also just describe what you want ("audit Connect for shipping") — the agent will pick a relevant Power on its own.</li>
  </ul>
</div>

<div class="coder-help-section">
  <div class="coder-help-heading">Subagents — parallel read-only helpers</div>
  <ul class="coder-help-list">
    <li>Ask for things like "audit this subsystem for shipping", "have subagents review X", or "search the codebase for every place that calls Y" and the agent will fan out read-only subagents in parallel.</li>
    <li>Built-in roles: <code>explore</code> (find every site matching a query), <code>plan</code> (design pass on a focused subtask), <code>review</code> (independent code review), <code>security_review</code> (disproof-oriented), <code>threat_model</code>, <code>research</code> (doc_search + doc_fetch with citations), <code>audit_zone</code> (one zone of a multi-agent subsystem audit).</li>
    <li>None of them can edit — they survey and report back so the lead agent can apply fixes.</li>
  </ul>
</div>

<div class="coder-help-section">
  <div class="coder-help-heading">Tips</div>
  <ul class="coder-help-list">
    <li>For build / run / deploy tasks, name the explicit command ("run <code>docker build -t x .</code>") instead of saying "run it". Reduces the chance the agent loops on discovery.</li>
    <li>On a small project, the whole codebase is already in the agent's context (project digest). No need to say "read my files first".</li>
    <li>Press <kbd>Esc</kbd> while the agent is running to cancel the current turn.</li>
  </ul>
</div>
`.trim();

function _formatTokenCount(n) {
  const value = Number(n || 0);
  if (!Number.isFinite(value) || value <= 0) return '0';
  return value.toLocaleString();
}

function _formatTokenBudget(tokens) {
  if (!tokens) return '';
  const count = _formatTokenCount(tokens.tokens);
  const limit = _formatTokenCount(tokens.limit);
  const pct = tokens.limit ? Math.round((Number(tokens.tokens || 0) / Number(tokens.limit || 1)) * 100) : 0;
  return `${count}/${limit} tokens (${pct}%)`;
}

/**
 * Compact older conversation turns while preserving the latest working set.
 *
 * Unlike /clear, this keeps the visible thread and LLM memory useful:
 * the server replaces the middle of the saved conversation with one
 * deterministic assistant summary, then the client reloads that compacted
 * message list and rebuilds currentCoder().chatHistory from it.
 */
async function _compactCoderConversation() {
  if (!currentCoder().workspaceId || !currentCoder().conversation) return;
  _stopActiveCoderRun('slash_compact');
  const beforeMessages = currentCoder().conversation.getHistory();
  const resp = await fetch(
    `/api/coder/conversation/${encodeURIComponent(currentCoder().workspaceId)}/compact`,
    {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: beforeMessages,
        keep_recent: 12,
        force: true,
        model: app.state.currentModel || '',
      }),
    },
  );
  if (!resp.ok) {
    const err = await resp.text().catch(() => resp.statusText);
    throw new Error(err || 'Compaction failed');
  }
  const data = await resp.json();
  if (Array.isArray(data.messages)) {
    currentCoder().conversation.loadHistory(data.messages);
    currentCoder().chatHistory = currentCoder().conversation.getMessagesForLLM();
  }
  _refreshRewindAffordance();
  const t = data.tokens || {};
  const detail = data.compacted
    ? `compacted ${_formatTokenCount(t.tokens_before)} → ${_formatTokenCount(t.tokens_after)} tokens`
    : `already compact (${_formatTokenCount(t.tokens_before)} tokens)`;
  _updateStatus('idle', detail);
  showToast?.(detail, data.compacted ? 'success' : 'info');
}

/**
 * Wipe the active workspace's conversation on server + client.
 *
 * Server-side DELETE clears the persisted message log, plan, plan
 * steps, tasks, and per-turn scratchpads. Files in the workspace
 * container are untouched — "/clear" is for the message history, not
 * the working copy. Client-side we reset the conversation renderer
 * and the in-memory LLM-context buffer so the next send starts clean.
 */
async function _clearCoderConversation() {
  if (!currentCoder().workspaceId) return;
  // Cancel any active stream so we don't race the clear. /clear
  // explicitly drops history, so we want the agent stopped server-
  // side too — otherwise it'd keep writing to a wiped conversation.
  _stopActiveCoderRun('slash_clear');
  try {
    await fetch(
      `/api/coder/conversation/${encodeURIComponent(currentCoder().workspaceId)}`,
      { method: 'DELETE', credentials: 'same-origin' },
    );
  } catch {
    // Silent — local clear still proceeds so the user gets the
    // "pressed /clear, something happened" feedback even if the
    // network call fails.
  }
  currentCoder().conversation?.clear();
  currentCoder().chatHistory = [];
  _updateStatus('idle', 'cleared');
  // Restore a minimal empty-state line + the full cheat-sheet as a
  // separate card below. Nesting the cheat-sheet inside
  // ``.coder-conv-empty`` (which is centered + column-flex) would
  // squash the bullet lists; keep the prompt line there and render
  // the sheet as a sibling using the same styles ``/help`` uses.
  const host = document.getElementById('coder-conv-messages');
  if (host && !document.getElementById('coder-conv-empty')) {
    const placeholder = document.createElement('div');
    placeholder.id = 'coder-conv-empty';
    placeholder.className = 'coder-conv-empty';
    placeholder.innerHTML = (
      '<p>Conversation cleared — ready for a fresh turn.</p>'
      + '<p class="coder-conv-empty-hint">Type <code>/help</code> '
      + 'below for a cheat-sheet, or just describe what you want.</p>'
    );
    host.appendChild(placeholder);
  }
}

// ---------------------------------------------------------------------------
// Conversation Agent — structured rendering in conversation pane
// ---------------------------------------------------------------------------

/**
 * Build the callbacks object passed to ``new CoderStream(...)``.
 *
 * Extracted so the live-turn path (``_runAgentInConversation``)
 * and the reattach path (``_attemptCoderReconnect``) share identical
 * UI wiring — tool cards, mission panel, status pills, the lot. The
 * server emits the same chunk format whether you're streaming a
 * fresh turn or replaying from the ledger; the only divergence is
 * the source (broker vs persistent table), and that lives in the
 * route, not here.
 *
 * ``thinkingEl`` is a closure-scoped mutable so onThinking can
 * append to the SAME element across multiple deltas, and reset
 * when a tool call interrupts the reasoning stream.
 */
function _buildCoderStreamCallbacks() {
  let thinkingEl = null;
  // Live reasoning block (reasoning_delta chunks). Closure-scoped like
  // thinkingEl; finalized (collapsed to "Reasoned for Xs") whenever the
  // model moves on — tool call, prose, retry, completion, error — so a
  // turn with several think→act cycles renders one tidy card per cycle.
  let reasoningBlock = null;
  const _closeReasoning = (opts) => {
    if (reasoningBlock) {
      currentCoder().conversation.finalizeReasoning(reasoningBlock, opts);
      reasoningBlock = null;
    }
  };

  // Helper: echo a one-liner to the terminal so split-view users
  // can see agent activity without switching panels.
  const _termEcho = (ansi) => {
    if (currentCoder().terminalId) Terminal.write(currentCoder().terminalId, ansi);
  };
  const _renderToolResult = (result) => _renderToolResultTo(_termEcho, result);

  return {
    onContent(text) {
      _closeReasoning();
      currentCoder().conversation.appendContent(text);
      _coderRecordDelta(text);
    },

    onThinking(text) {
      if (!thinkingEl) thinkingEl = currentCoder().conversation.addThinking();
      currentCoder().conversation.appendThinking(thinkingEl, text);
      _coderRecordDelta(text);
    },

    onReasoning(text) {
      // Not recorded via _coderRecordDelta: reasoning is ephemeral
      // peek-at-it context (same contract as the backend, which keeps
      // it out of the replay ledger), not turn content.
      if (!reasoningBlock) reasoningBlock = currentCoder().conversation.addReasoning();
      currentCoder().conversation.appendReasoning(reasoningBlock, text);
    },

    // Backend Stage lifecycle (model_load / model_swap / slot_restore /
    // prefill). Mirrors chat's stage_* handling — drives a real
    // progress bar in the coder status line so the user can see what
    // the model is doing during 30-180s prefills instead of staring at
    // "waiting for model…" with no signal.
    onStage(payload) {
      _coderStageEvent(payload);
    },

    onToolCall(id, tool, input) {
      thinkingEl = null;
      _closeReasoning();
      currentCoder().conversation.addToolCall(id, tool, input);

      let desc = tool;
      if (tool === 'shell_exec') desc = `$ ${input.command || ''}`;
      else if (tool === 'file_read') desc = `reading ${input.path || ''}`;
      else if (tool === 'code_edit' || tool === 'file_write') desc = `editing ${input.path || ''}`;
      else if (tool === 'code_grep') desc = `searching "${input.pattern || ''}"`;
      _updateStatus('executing', desc);

      // Terminal echo — one-line tool indicator
      _termEcho(`\x1b[33m ┄ ${desc}\x1b[0m\r\n`);
    },

    onToolResult(id, result) {
      currentCoder().conversation.updateToolResult(id, result);
      _renderToolResult(result);
      // Persist the partial turn incrementally so an interruption after this
      // tool call (network drop, user leaves/returns) survives — debounced so
      // a flurry of results is one whole-history write, not one per result.
      _saveConversationSoon(currentCoder().workspaceId);
      if (result?.tool === 'publish_ports' && result?.metadata?.workspace_recreated) {
        void _reconnectActiveTerminal().then(() => _refreshPorts()).catch(() => {});
      }
    },

    onShellOutput(text) {
      currentCoder().conversation.appendShellOutput(text);
    },

    // Soft stall hint (progress over abort). Fires only after 45s of
    // total silence outside a known load/prefill stage. No abort button —
    // the agent auto-retries and there's nothing for the user to cancel.
    // The next real event (token, tool, status) repaints the detail, so
    // we only need to act on the stalled=true edge; the agent is still
    // executing, so the status state and the stop affordance stay put.
    onStall(stalled) {
      if (stalled) {
        _updateStatus('executing', 'still working — slower than usual…');
      }
    },

    onSubagentProgress(progress) {
      // Live inner-loop event from a running subagent. The
      // conversation finds the active task_dispatch card by
      // instance_id and appends one activity row + updates the
      // running stats line. See coder-conversation.js::
      // updateSubagentProgress for the binding + DOM logic.
      currentCoder().conversation.updateSubagentProgress?.(progress);
    },

    onStepStart(step, total, description) {
      currentCoder().conversation.addStep(step, total, description);
      _updateStatus('executing', `step ${step}/${total}`);
      _termEcho(`\x1b[36m ● Step ${step}/${total}: ${description}\x1b[0m\r\n`);
    },

    onStrategy(strategy) {
      const labels = {
        react: 'Multi-step (observe & adapt)',
        mission: 'Mission (verified steps)',
        direct: 'One-shot',
        decompose: 'Decompose',
        architect: 'Architect',
      };
      _updateStatus('executing', labels[strategy] || strategy);
      _termEcho(`\x1b[36m ● Strategy: ${labels[strategy] || strategy}\x1b[0m\r\n`);
    },

    async onPowerActivated(payload) {
      _setRuntimePowerActivation(payload);
      try {
        const mod = await import('./powers.js');
        _appendCoderEventHtml(mod.powerActivationHtml(payload));
      } catch { /* ignore */ }
      const checkpoint = payload?.checkpoint ? ` @ ${payload.checkpoint}` : '';
      _termEcho(`\x1b[36m ● Power: ${(payload?.display_name || payload?.id || 'unknown')}${checkpoint}\x1b[0m\r\n`);
    },

    // ── Mission runtime events ─────────────────────────────────────
    onMissionStarted(mission) {
      currentCoder().missionPanel?.start(mission);
      _termEcho(`\x1b[36m ● Mission: ${mission.length} step${mission.length === 1 ? '' : 's'}\x1b[0m\r\n`);
    },
    onPromiseStarted(promise) {
      currentCoder().missionPanel?.onPromiseStarted(promise);
      _termEcho(`\x1b[33m ▸ ${promise.description}\x1b[0m\r\n`);
    },
    onPromiseVerifying(promise) {
      currentCoder().missionPanel?.onPromiseVerifying(promise);
    },
    onPromiseFulfilled(promise) {
      currentCoder().missionPanel?.onPromiseFulfilled(promise);
      _termEcho(`\x1b[32m ✓ ${promise.description}\x1b[0m\r\n`);
    },
    onPromiseRetry(promise, reason) {
      currentCoder().missionPanel?.onPromiseRetry(promise, reason);
      _termEcho(`\x1b[33m ⟳ Retry: ${(reason || '').slice(0, 80)}\x1b[0m\r\n`);
    },
    onPromiseRejected(promise, reason) {
      currentCoder().missionPanel?.onPromiseRejected(promise, reason);
      _termEcho(`\x1b[31m ✖ ${promise.description} — ${(reason || '').slice(0, 80)}\x1b[0m\r\n`);
    },
    onPromiseDecomposed(promise, children) {
      currentCoder().missionPanel?.onPromiseDecomposed(promise, children);
      _termEcho(`\x1b[36m ⤷ Decomposed into ${children.length} sub-step${children.length === 1 ? '' : 's'}\x1b[0m\r\n`);
    },
    onMissionCompleted(data) {
      currentCoder().missionPanel?.onMissionCompleted(data);
      _termEcho(`\x1b[32m ━━━ Mission complete ━━━\x1b[0m\r\n`);
    },
    onMissionFailed(data) {
      currentCoder().missionPanel?.onMissionFailed(data);
      const reason = data?.first_reason || data?.reason || '';
      _termEcho(`\x1b[31m ━━━ Mission failed${reason ? ': ' + reason.slice(0, 80) : ''} ━━━\x1b[0m\r\n`);
    },
    onRateLimited(info) {
      currentCoder().missionPanel?.onRateLimited(info);
      _updateStatus('executing', `paused — ${info.reason || 'rate limited'} (${info.waitSeconds}s)`);
      _termEcho(
        `\x1b[33m ⏸ Paused ${info.waitSeconds}s — ${info.reason || 'rate limited'} ` +
        `(retry ${info.attempt}/${info.maxRetries})\x1b[0m\r\n`,
      );
    },

    async onReviewPending(turnId) {
      // Mount the review panel directly into the conversation scroll
      // so it appears inline after the turn's synthesis message. The
      // panel fetches the bundle, renders itself collapsed, and wires
      // its own accept/reject interactions. Failures are silent —
      // backend off or bundle gone is benign.
      const container = document.getElementById('coder-conv-messages');
      if (!container) return;
      try {
        const { mountReviewPanel } = await import('./coder-review.js');
        await mountReviewPanel(turnId, container);
      } catch (err) {
        // Dynamic import failure means the review module is broken —
        // the turn itself still ran fine. Log but don't surface.
        console.warn('coder-review: mount failed', err);
      }

      // Post-review refresh: when the user accepts/rejects/partials,
      // disk may have been rolled back by the restore path OR
      // committed by the accept path. Refresh the file tree so the
      // sidebar reflects actual state. The listener is registered
      // once per session — idempotent via the currentCoder().reviewListenerWired
      // guard below.
      if (!currentCoder().reviewListenerWired) {
        document.addEventListener('coder:turn-reviewed', () => {
          _scheduleFileTreeRefresh();
          _startGitPolling();
        });
        currentCoder().reviewListenerWired = true;
      }
    },

    onRunDetails(runId) {
      _setActiveRunDetails(runId);
    },

    onStatus(phase, status, aug = {}) {
      // Cooperative events — steer-delivered ack, queue-followup
      // auto-chain. _handleCoopChunk is defensive at its own
      // boundary (null-guarded everywhere) so callers don't need a
      // try/catch wrapper. No-op on unrelated statuses, so it's
      // safe to invoke unconditionally.
      _handleCoopChunk(aug);
      // Permanent backend failure (4xx auth / validation / not-found).
      // Retrying the same payload hits the same wall, so surface the
      // provider's actual reply text in a pill with Dismiss-only — the
      // user needs the detail to fix the request (wrong model, missing
      // tool name, message-ordering bug, expired key, etc.). Falls back
      // to the bare status update if the chunk had no error_message
      // (older backends or unclassified errors).
      if (status === 'error') {
        const errMsg = (aug?.error_message || '').trim();
        if (errMsg) {
          const code = aug?.retry_status_code ? ` (HTTP ${aug.retry_status_code})` : '';
          // Quota errors phrase differently — it's not "your request is
          // broken", it's "you hit a per-minute cap; wait or switch model".
          // Keeps the permanent-style pill (no Try Again, since the
          // window won't reset in time) but with quota-aware copy.
          const isQuota = aug?.error_kind === 'quota';
          const title = isQuota
            ? `Provider quota exceeded${code}`
            : `Provider rejected request${code}`;
          currentCoder().conversation?.addRecoverableError(errMsg, {
            title,
            permanent: true,
            onDismiss: () => _updateStatus('idle', 'stopped'),
          });
          _updateStatus('error', isQuota
            ? `quota exceeded${code} — wait or switch model`
            : `provider error${code}`);
        } else {
          _updateStatus('error', 'agent error');
        }
      }
      // Transient backend failure exhausted its retry budget (429 / 5xx
      // / network blip). Show a Try Again / Stop pill in the conversation
      // — the same prompt is likely to succeed if the user retries.
      else if (status === 'recoverable_error') {
        const code = aug?.retry_status_code ? ` (HTTP ${aug.retry_status_code})` : '';
        const providerMsg = (aug?.error_message || '').trim();
        const detail = providerMsg
          ? `${providerMsg} — the next attempt may succeed.`
          : (aug?.error_kind === 'transient'
            ? `The model provider was slow or overloaded${code}. Your prompt didn't complete — the next attempt is likely to succeed.`
            : `The provider rejected the request${code}. Try a different model or check the model's settings.`);
        currentCoder().conversation?.addRecoverableError(detail, {
          title: aug?.retry_status_code === 429
            ? 'Provider rate-limited'
            : 'Backend timeout',
          onRetry: () => {
            if (currentCoder().lastAgentRequest) {
              _runAgentInConversation(
                currentCoder().lastAgentRequest.request,
                currentCoder().lastAgentRequest.attachments,
              );
            }
          },
          onDismiss: () => _updateStatus('idle', 'stopped'),
        });
        _updateStatus('error', 'backend timeout — try again');
      }
      // Stagnation detector tripped — the main model was looping on the
      // same failure (or making no progress) and the workspace has a
      // heavyweight model configured. Remaining iterations of THIS
      // TURN run on the buddy. Surface this loud and clear: the user is
      // now spending paid tokens via the heavyweight model and should
      // see it the moment it happens.
      else if (status === 'escalated_to_buddy') {
        const buddy = aug?.buddy || 'heavyweight model';
        const reason = aug?.reason === 'repeated_validation_error'
          ? 'repeated tool errors'
          : aug?.reason === 'no_progress'
            ? 'no progress for 3 iterations'
            : 'stagnation detected';
        _updateStatus('executing', `⚡ escalated to ${buddy}`);
        _termEcho(`\x1b[35m ⚡ Escalated to ${buddy} — ${reason}\x1b[0m\r\n`);
      }
      // Mid-backoff hint — the in-stream retry is in progress; let the
      // user know it's not stuck, just waiting for the provider to
      // recover. Both status bar AND terminal echo so the signal
      // shows wherever the user is looking.
      else if (status === 'retrying') {
        // The attempt that produced the current reasoning stream died —
        // close its block as interrupted; the retry opens a fresh one.
        _closeReasoning({ interrupted: true });
        const waitS = Math.round(aug?.retry_wait_s || 5);
        const attempt = aug?.retry_attempt || 2;
        const maxAttempts = aug?.retry_max || 3;
        const codeBit = aug?.retry_status_code ? ` (${aug.retry_status_code})` : '';
        _updateStatus('executing',
          `provider slow${codeBit} — retry ${attempt}/${maxAttempts} in ${waitS}s…`);
        _termEcho(`\x1b[33m ⟳ Provider slow${codeBit} — retrying in ${waitS}s (attempt ${attempt}/${maxAttempts})\x1b[0m\r\n`);
      }
      else if (status === 'budget' && aug.tokens) {
        _updateStatus('executing', _formatTokenBudget(aug.tokens));
      }
      else if (status === 'compaction') {
        const before = _formatTokenCount(aug.tokens_before);
        const after = _formatTokenCount(aug.tokens_after);
        _updateStatus('executing', `compacted ${before} → ${after} tokens`);
      }
      else if (status === 'fixing') {
        _updateStatus('executing', 'fixing errors...');
        _termEcho(`\x1b[33m ⟳ Retrying...\x1b[0m\r\n`);
      }
      else if (status === 'max_iterations_reached') _updateStatus('executing', 'iteration limit reached');
      else if (status === 'repeat_stopped') _updateStatus('error', 'stopped: repeated action');
      // Streaming sub-states — label the dead-air windows so the user
      // sees that work is happening even when no tokens have arrived
      // yet. Phase-scoped so "planning" vs "executing" context shows.
      else if (status === 'awaiting_first_token') {
        const label = phase === 'planning' ? 'planning — waiting for model…'
                    : phase === 'executing' ? 'waiting for model…'
                    : 'waiting for model…';
        _updateStatus('executing', label);
      }
      else if (status === 'thinking') {
        const label = phase === 'planning' ? 'planning — reasoning…'
                    : 'reasoning…';
        _updateStatus('executing', label);
        _coderStartStreamTracker(label);
      }
      else if (status === 'responding') {
        const label = phase === 'planning' ? 'planning — writing…'
                    : 'writing…';
        _updateStatus('executing', label);
        _coderStartStreamTracker(label);
      }
    },

    onComplete(fullResponse) {
      _closeReasoning();
      currentCoder().conversation.finalizeResponse();
      if (fullResponse) {
        currentCoder().chatHistory.push({ role: 'assistant', content: fullResponse });
      }
      _refreshRewindAffordance();
      _setRuntimePowerActivation(null);
      _updateStatus('idle');
      _updateIntentBar();
      _populateFileTree(currentCoder().workspaceId);
      _saveConversation(currentCoder().workspaceId);
      // Cooperative auto-chain: if a ``queue_followup`` chunk landed
      // during this turn, dispatch the queued content as a fresh
      // turn now that the stream has fully closed. Deferred from the
      // onStatus boundary because at that point currentCoder().coderStream is
      // still .isActive() and the cooperative-mode router would
      // re-queue into the inbox infinitely. See ``_handleCoopChunk``
      // for the rationale.
      _flushPendingCoopChain();
      // Any bubble still "queued"/"steering" after the flush missed
      // its resolution chunk — unstick it (see _sweepStaleCoopBubbles).
      _sweepStaleCoopBubbles('turn_ended');
      if (currentCoder().terminalId) Terminal.fit(currentCoder().terminalId);
      _termEcho(`\x1b[90m ━━━ Done ━━━\x1b[0m\r\n`);
      _coderProgressClear();
      // Audible "it's ready" cue if the user opted in and stepped away.
      _maybeChimeOnDone();
      // The agent task has reported completion. Drop the persisted
      // active-run pointer so a hard reload doesn't try to reattach
      // to a finished run.
      _clearActiveRunDetails();
    },

    onError(error) {
      _closeReasoning({ interrupted: true });
      _coderProgressClear();
      // Capture the partial turn so a Resume can let the model "know the
      // situation": finalize the streamed (partial) assistant text into the
      // conversation, then rebuild currentCoder().chatHistory from it (the canonical
      // serializer — same as a conversation switch). onComplete does this on
      // success; this is its failure-path twin. Without it a mid-flight
      // network death drops everything the agent had done this turn.
      currentCoder().conversation.finalizeResponse();
      currentCoder().chatHistory = currentCoder().conversation.getMessagesForLLM();
      // A stream/network death (NOT a backend 'recoverable_error' status —
      // that path has its own Try-Again pill). Offer Resume: re-enter with an
      // interruption-aware instruction so the agent re-checks state and
      // continues where it left off, instead of the user retyping. Falls back
      // to a bare error when there's nothing in-flight to resume.
      if (currentCoder().workspaceId && currentCoder().lastAgentRequest) {
        currentCoder().conversation.addRecoverableError(
          'The connection dropped mid-task. Resume to have the agent re-check '
          + 'what it had started and continue from where it left off.',
          {
            title: 'Turn interrupted',
            retryLabel: 'Resume',
            busyLabel: 'Resuming…',
            dismissLabel: 'Dismiss',
            // Fire-and-forget so the pill dismisses as the new turn starts
            // streaming (matches the Try-Again pattern); the resumed turn
            // renders into the conversation itself.
            onRetry: () => { void _resumeInterruptedTurn(); },
          },
        );
      } else {
        currentCoder().conversation.addError(error);
      }
      // Persist the interrupted turn (prompt + tool calls + the error marker)
      // so a mid-flight failure isn't erased on reload — onComplete saves on
      // success; this is its failure-path counterpart. Flush immediately
      // (not debounced) since the run is over.
      void _saveConversation(currentCoder().workspaceId);
      _setRuntimePowerActivation(null);
      _updateStatus('error', error);
      _updateIntentBar();
      _termEcho(`\x1b[31m ✖ ${error}\x1b[0m\r\n`);
      // The run ended (even if it failed) — cue the user back if they
      // opted in and stepped away.
      _maybeChimeOnDone();
      // Coop hygiene on error (matches onComplete's flush behaviour):
      //  1. If a queue_followup chunk landed BEFORE the error, the
      //     chained turn was stashed for onComplete dispatch. onError
      //     would orphan it — user sees their queued bubble "delivered"
      //     but no turn fires. Flush here so the user's intent isn't
      //     swallowed by an upstream blip.
      //  2. Reset _suppressNextUserBubble so a stale "skip the next
      //     addUserMessage" flag from a partially-completed dispatch
      //     doesn't bleed into the user's NEXT legitimate message and
      //     silently drop its bubble.
      _flushPendingCoopChain();
      _suppressNextUserBubble = false;
      // Same missed-chunk sweep as onComplete — an errored/cancelled
      // stream is the MOST likely to have lost its queue_dropped chunk.
      _sweepStaleCoopBubbles('turn_interrupted');
    },
  };
}


// The instruction sent when the user clicks Resume on an interrupted turn.
// The partial turn is already in currentCoder().chatHistory (captured by onError), so the
// model sees what it had done; the backend rebuilds the workspace snapshot
// each turn, so it also sees the real current file state. This just tells it
// NOT to trust that in-progress work finished, and to carry on.
const _RESUME_INTERRUPTED_PROMPT =
  '↻ Resume: the previous turn was cut off by a connection error before it '
  + 'finished. Do not assume any in-progress edit or command completed — '
  + 're-check the current state of the files and commands you were working on, '
  + 'then continue from where you left off to finish my original request.';

// Re-enter an interrupted coder turn with interruption context. Mirrors a
// normal send; the only difference is the canned continue-aware instruction.
async function _resumeInterruptedTurn() {
  if (!currentCoder().workspaceId || !currentCoder().conversation) return;
  await _runAgentInConversation(_RESUME_INTERRUPTED_PROMPT, []);
}


async function _runAgentInConversation(request, attachments = []) {
  if (!currentCoder().workspaceId || !currentCoder().conversation) return;

  // Capture for the recoverable-error pill's Try Again button — if the
  // backend exhausts its transient-retry budget mid-turn, the user can
  // re-fire this exact prompt without retyping.
  currentCoder().lastAgentRequest = { request, attachments };

  // Cancel any active stream — new turn replaces old. The workspace
  // only runs one coder turn at a time, so this is also a cancel-
  // server-side gesture, not just a fetch close.
  _stopActiveCoderRun('new_turn_started');

  _updateStatus('executing', 'thinking...');
  _setIntentButtons([{ label: 'Cancel (Esc)', action: '__cancel' }]);

  // Build the outgoing user-message payload with attachments merged
  // in: images land on msg.images (consumed by the chat_images
  // pipeline → VL models see them inline); non-image files add a
  // "📎 Attached: <path>" footer so the agent can file_read them.
  //
  // Two text variants flow from here:
  //   - ``request`` (original user prompt) → rendered in the chat
  //     bubble, stays clean. Attachments are visualised as inline
  //     chips beside the text via addUserMessage's second arg.
  //   - ``payload.content`` (footer-augmented) → sent to the LLM so
  //     the agent sees both an actionable prompt and the file paths.
  const { buildMessagePayload } = await import('./coder-attachments.js');
  const payload = buildMessagePayload(request, attachments);
  const userText = payload.content;

  // Render in conversation with CLEAN text + visual attachment chips.
  // The LLM still gets the footer via payload.content below, but the
  // UI stays readable.
  //
  // _suppressNextUserBubble is set by callers that already rendered
  // a bubble for THIS exact message — the coop auto-chain (after a
  // queue_followup drain) and the 409 interject fallback. Skipping
  // addUserMessage here is the difference between the user seeing
  // one bubble (correct) vs. two duplicate bubbles for one intent.
  if (_suppressNextUserBubble) {
    _suppressNextUserBubble = false;
  } else {
    currentCoder().conversation.addUserMessage(request, attachments);
  }
  _refreshRewindAffordance();
  // Persist the prompt immediately so that even a turn that dies before its
  // first tool result (instant network failure) leaves a record the user can
  // return to and continue from — not a reverted, prompt-less conversation.
  void _saveConversation(currentCoder().workspaceId);

  // Capture terminal context BEFORE echoing the new request banner.
  // Otherwise the model sees its own just-written prompt/status lines
  // in [Terminal context], and long wrapped requests get re-fed in a
  // split/malformed shape.
  const scrollback = currentCoder().terminalId ? Terminal.getScrollback(currentCoder().terminalId, 20) : '';

  // Echo to terminal so split-view users see what was asked — use the
  // original user request (not the footer-augmented version) so the
  // terminal doesn't show "📎 Attached: /path" noise.
  if (currentCoder().terminalId) {
    Terminal.write(currentCoder().terminalId,
      `\r\n\x1b[1;36m// ${request}\x1b[0m\r\n\x1b[90m━━━ Agent working... ━━━\x1b[0m\r\n`);
  }

  // Build messages for LLM context (conversation history + terminal context)
  const contextText = scrollback.trim()
    ? `[Terminal context]\n${scrollback}\n[/Terminal context]\n\n${userText}`
    : userText;

  // Push with images if any — the /api/chat pipeline's
  // resolve_chat_image_urls hook expands chat-image URLs to base64
  // data URLs before the model sees them.
  const historyEntry = { role: 'user', content: contextText };
  if (payload.images?.length) historyEntry.images = payload.images;
  currentCoder().chatHistory.push(historyEntry);
  if (currentCoder().chatHistory.length > 30) currentCoder().chatHistory = currentCoder().chatHistory.slice(-30);

  _setRuntimePowerActivation(null);
  currentCoder().coderStream = new CoderStream(_buildCoderStreamCallbacks());

  try {
    // Honor the coder thinking toggle for this turn. Only set the
    // kwarg when explicitly enabled — leaving the field unset keeps
    // the per-strategy default behavior (native = enable_thinking
    // false; canonical/hybrid = template default). Sending null
    // would override and force-disable, which isn't what we want.
    const thinkingKwargs = _isCoderThinkingEnabled()
      ? { enable_thinking: true }
      : null;
    await currentCoder().coderStream.send({
      model: app.state.currentModel || '',
      messages: currentCoder().chatHistory,
      workspaceId: currentCoder().workspaceId,
      chatTemplateKwargs: thinkingKwargs,
    });
  } catch (err) {
    currentCoder().conversation.addError(err.message || 'Request failed');
    _updateStatus('error', err.message);
  }
}

// ---------------------------------------------------------------------------
// Inline Agent — runs from // prefix in terminal
// ---------------------------------------------------------------------------

async function _runAgentInTerminal(request) {
  if (!currentCoder().terminalId || !currentCoder().workspaceId) return;

  if (currentCoder().terminalAgentAbort) currentCoder().terminalAgentAbort.abort();
  currentCoder().terminalAgentAbort = new AbortController();

  // Block terminal input during agent execution
  Terminal.setAgentActive(currentCoder().terminalId, true);

  // Update status + show execution-mode intent bar. Button label
  // mentions Ctrl+C because that's the more intuitive cancel key
  // (Escape still works — see terminal.js). The banner line below
  // reinforces in-terminal so users who ignore the intent bar also
  // learn about it.
  _updateStatus('executing', 'thinking...');
  _setIntentButtons([
    {label: 'Cancel (Ctrl+C)', action: '__cancel'},
  ]);

  // Snapshot terminal context BEFORE writing the new request. Feeding the
  // echoed request/banner back into the model duplicates the user's prompt
  // and can split long requests across wrapped rows.
  const scrollback = Terminal.getScrollback(currentCoder().terminalId, 40);

  // Show the request
  Terminal.write(currentCoder().terminalId,
    `\x1b[1;36m// ${request}\x1b[0m\r\n`);  // Cyan bold
  Terminal.write(currentCoder().terminalId,
    `\x1b[90m━━━ Agent thinking... ━━━ \x1b[2m(Ctrl+C or Esc to cancel)\x1b[0m\r\n`);

  // Add to chat history for context continuity
  const contextText = scrollback.trim()
    ? `[Terminal context]\n${scrollback}\n[/Terminal context]\n\n${request}`
    : request;
  currentCoder().chatHistory.push({ role: 'user', content: contextText });

  // Window chat history to prevent token overflow (keep last 30 messages)
  const MAX_CHAT_HISTORY = 30;
  if (currentCoder().chatHistory.length > MAX_CHAT_HISTORY) {
    currentCoder().chatHistory = currentCoder().chatHistory.slice(-MAX_CHAT_HISTORY);
  }

  let fullResponse = '';

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Augmentum-Mode': 'coder',
        'X-Augmentum-Workspace': currentCoder().workspaceId,
      },
      body: JSON.stringify({
        model: app.state.currentModel || '',
        messages: currentCoder().chatHistory,
        stream: true,
      }),
      signal: currentCoder().terminalAgentAbort.signal,
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      const lines = buf.split('\n');
      buf = lines.pop() || '';

      for (const line of lines) {
        if (!line.trim() || line === 'data: [DONE]') continue;
        const jsonStr = line.startsWith('data: ') ? line.slice(6) : line;
        try {
          const chunk = JSON.parse(jsonStr);

          // Content delta — render inline in terminal
          const delta = chunk.message?.content || chunk.choices?.[0]?.delta?.content || '';
          if (delta) {
            fullResponse += delta;
            // Filter out tool call JSON and completion tags from terminal display
            let display = stripToolCallJSON(delta.replace(/<task_complete\/>/g, ''))
              .replace(/\n{3,}/g, '\n');  // collapse excess newlines from removed JSON
            if (display.trim()) {
              Terminal.write(currentCoder().terminalId, display.replace(/\n/g, '\r\n'));
            }
          }

          // Tool activity
          const aug = chunk.augmentum || chunk.choices?.[0]?.delta?.augmentum;
          if (aug && aug.status === 'tool_call') {
            const tc = aug.tool_call || {};
            const toolName = tc.tool || tc.name || 'tool';
            const toolInput = tc.input || {};
            // Strip the /workspace/ prefix so the tool-call line reads
            // clean; users already know files live there.
            const shortPath = (p) => (p || '').replace(/^\/workspace\//, '');
            let label = toolName;
            let target = '';
            if (toolName === 'shell_exec') { label = '$'; target = toolInput.command || ''; }
            else if (toolName === 'shell_read') { label = '$'; target = toolInput.command || ''; }
            else if (toolName === 'file_read') { label = 'read'; target = shortPath(toolInput.path); }
            else if (toolName === 'file_write') { label = 'write'; target = shortPath(toolInput.path); }
            else if (toolName === 'code_edit' || toolName === 'code_edit_batch') { label = 'edit'; target = shortPath(toolInput.path); }
            else if (toolName === 'file_list') { label = 'list'; target = shortPath(toolInput.path); }
            else if (toolName === 'find_files') { label = 'glob'; target = toolInput.pattern || ''; }
            else if (toolName === 'code_grep') { label = 'grep'; target = `"${toolInput.pattern || ''}"`; }
            else if (toolName === 'code_search') { label = 'search'; target = toolInput.query || ''; }
            else if (toolName === 'dir_tree') { label = 'tree'; target = shortPath(toolInput.path || ''); }
            else if (toolName === 'test_run') { label = 'test'; target = toolInput.command || ''; }

            _updateStatus('executing', target ? `${label} ${target}` : label);

            // New visual format: ▸ label target
            //   dim chevron + bold-yellow label + dim-cyan target.
            // Replaces the pre-2026-04-21 "┄┄┄ desc ┄┄┄" banner,
            // which added visual weight without conveying structure.
            // The indented ▸ acts as a section gutter that pairs with
            // the 3-space indent on result lines in _renderToolResultTo.
            const ANSI_DIM = '\x1b[90m';
            const ANSI_BOLD_YLW = '\x1b[1;33m';
            const ANSI_CYAN = '\x1b[36m';
            const ANSI_RESET = '\x1b[0m';
            const line = target
              ? `\r\n${ANSI_DIM}▸ ${ANSI_RESET}${ANSI_BOLD_YLW}${label}${ANSI_RESET} ${ANSI_CYAN}${target}${ANSI_RESET}\r\n`
              : `\r\n${ANSI_DIM}▸ ${ANSI_RESET}${ANSI_BOLD_YLW}${label}${ANSI_RESET}\r\n`;
            Terminal.write(currentCoder().terminalId, line);
          }

          if (aug && aug.status === 'tool_result') {
            // Route through the shared renderer so this path (//-prefix
            // terminal agent) and the conversation-driven path share
            // identical tool-result semantics. Pre-2026-04-20 this
            // block had its own 400-char clipped, reads-and-shells-only
            // rendering that left file_write / test_run / task_list
            // completely silent on success.
            _renderToolResultTo(
              (ansi) => Terminal.write(currentCoder().terminalId, ansi),
              aug.tool_result || {},
            );
            if (aug.tool_result?.tool === 'publish_ports' && aug.tool_result?.metadata?.workspace_recreated) {
              void _reconnectActiveTerminal().then(() => _refreshPorts()).catch(() => {});
            }
          }

          // Fallback-summary chunk at turn end (see handler.py's
          // _render_fallback_summary). content_delta is already
          // streaming via the main delta path above — nothing else to
          // do here, but we capture the meta event so the UI can tag
          // the final banner appropriately if we want to later.

          // Strategy / phase status indicators. Only render on the
          // dedicated strategy-announce chunk (status=="strategy") — the
          // completion chunk (status=="complete") also carries a
          // ``strategy`` field for logging, which pre-2026-04-20 was
          // triggering a duplicate "Strategy: hybrid" line in the
          // terminal right before "Done".
          if (aug && aug.strategy && aug.status === 'strategy') {
            const labels = { react: 'Multi-step', direct: 'One-shot', decompose: 'Decompose', architect: 'Architect' };
            Terminal.write(currentCoder().terminalId,
              `\x1b[36m ● Strategy: ${labels[aug.strategy] || aug.strategy}\x1b[0m\r\n`);
          }
          if (aug && aug.status === 'fixing') {
            Terminal.write(currentCoder().terminalId,
              `\r\n\x1b[33m ⟳ Errors detected — retrying...\x1b[0m\r\n`);
          }
          if (aug && aug.status === 'step_start') {
            Terminal.write(currentCoder().terminalId,
              `\r\n\x1b[36m ● Step ${aug.step}/${aug.total}: ${aug.description || ''}\x1b[0m\r\n`);
          }
          if (aug && aug.status === 'repeat_stopped') {
            Terminal.write(currentCoder().terminalId,
              `\r\n\x1b[31m ✖ Stopped: repeated the same action\x1b[0m\r\n`);
          }
        } catch { /* partial JSON */ }
      }
    }

    if (fullResponse) {
      currentCoder().chatHistory.push({ role: 'assistant', content: fullResponse });
    }

    // Check if the agent asked a question — extract options for intent bar
    const isQuestion = fullResponse.includes('?') && !fullResponse.includes('━━━ Done');
    if (isQuestion) {
      const options = _extractQuestionOptions(fullResponse);
      if (options.length > 0) {
        _setIntentButtons(options.slice(0, 4).map(opt => ({
          label: opt,
          action: '//' + opt,
        })));
        // Don't update intent bar from terminal output — keep question options
        _updateStatus('idle', 'waiting for your choice');
      } else {
        _updateIntentBar();
      }
    } else {
      _scheduleFileTreeRefresh();
      if (_checkpointsExpanded) _loadCheckpoints();
    }

    // Mark open editor files as potentially modified
    currentCoder().editorFiles.forEach(f => {
      // Visual indication could be added to tabs here
    });

  } catch (err) {
    if (err.name !== 'AbortError') {
      Terminal.write(currentCoder().terminalId,
        `\r\n\x1b[31mAgent error: ${err.message || 'request failed'}\x1b[0m\r\n`);
      _updateStatus('error', err.message || 'request failed');
    }
  } finally {
    currentCoder().terminalAgentAbort = null;
    Terminal.setAgentActive(currentCoder().terminalId, false);

    // Don't show "Done" if the agent asked a question
    const askedQuestion = fullResponse.includes('?') && !fullResponse.includes('tool');
    if (!askedQuestion) {
      Terminal.write(currentCoder().terminalId, '\r\n\x1b[90m━━━ Done ━━━\x1b[0m\r\n');
      const summary = fullResponse.slice(0, 60).replace(/\n/g, ' ').trim();
      _updateStatus('idle', summary ? 'last: ' + summary + (fullResponse.length > 60 ? '...' : '') : '');
    }

    // Refresh intent bar based on new terminal output
    _updateIntentBar();

    // Background: update codebase index (agent may have created/modified files)
    if (currentCoder().workspaceId) {
      fetch(`/api/coder/index/${encodeURIComponent(currentCoder().workspaceId)}`, { method: 'POST' }).catch(() => {});
      _startIndexProgressPoll(currentCoder().workspaceId);
    }
  }
}

// ---------------------------------------------------------------------------
// Checkpoints (git auto-commit timeline)
// ---------------------------------------------------------------------------

let _checkpointsExpanded = false;
let _checkpointsInitWired = false;

async function _loadCheckpoints() {
  if (!currentCoder().workspaceId) return;
  const listEl = document.getElementById('coder-checkpoints-list');
  const countEl = document.getElementById('coder-checkpoints-count');
  if (!listEl) return;

  try {
    const resp = await fetch(`/api/coder/checkpoints/${encodeURIComponent(currentCoder().workspaceId)}?limit=20`);
    if (!resp.ok) return;
    const data = await resp.json();
    const checkpoints = data.checkpoints || [];

    if (countEl) countEl.textContent = checkpoints.length > 0 ? `(${checkpoints.length})` : '';
    // Mirror the count into the status-bar tile so it's discoverable
    // even when the left panel is collapsed/closed.
    const tileLabel = document.getElementById('coder-checkpoint-tile-label');
    const tile = document.getElementById('coder-checkpoint-tile');
    if (tileLabel) tileLabel.textContent = String(checkpoints.length);
    if (tile) tile.classList.toggle('hidden', checkpoints.length === 0);

    if (checkpoints.length === 0) {
      listEl.innerHTML = '<div class="coder-checkpoint-empty">No checkpoints yet</div>';
      return;
    }

    listEl.innerHTML = checkpoints.map((cp, i) => {
      const time = cp.timestamp ? new Date(cp.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
      const isCurrent = i === 0;
      return `<div class="coder-checkpoint-item ${isCurrent ? 'current' : ''}" data-hash="${escapeHtml(cp.hash)}" data-subject="${escapeHtml(cp.message)}">
        <span class="coder-checkpoint-dot"></span>
        <div class="coder-checkpoint-info" title="View what changed in this commit">
          <span class="coder-checkpoint-msg">${escapeHtml(cp.message)}</span>
          <span class="coder-checkpoint-meta">${escapeHtml(cp.hash)} · ${time}</span>
        </div>
        ${!isCurrent ? '<button class="coder-checkpoint-revert" title="Revert to this point">Revert</button>' : '<span class="coder-checkpoint-current">current</span>'}
      </div>`;
    }).join('');

    // Click a checkpoint's info to browse its diff (git show).
    listEl.querySelectorAll('.coder-checkpoint-info').forEach((info) => {
      info.addEventListener('click', () => {
        const item = info.closest('.coder-checkpoint-item');
        if (item) _showCommitDiffModal(item.dataset.hash, item.dataset.subject || '');
      });
    });

    // Wire revert buttons
    listEl.querySelectorAll('.coder-checkpoint-revert').forEach(btn => {
      btn.onclick = async () => {
        const hash = btn.closest('.coder-checkpoint-item').dataset.hash;
        if (!confirm(`Revert workspace to checkpoint ${hash}? Current changes will be preserved as a new checkpoint.`)) return;
        btn.textContent = 'Reverting...';
        btn.disabled = true;
        try {
          const resp = await fetch(`/api/coder/checkpoints/${encodeURIComponent(currentCoder().workspaceId)}/revert`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ hash }),
          });
          if (resp.ok) {
            showToast(`Reverted to ${hash}`, 'success');
            _loadCheckpoints();
            _populateFileTree(currentCoder().workspaceId);
          } else {
            showToast('Revert failed', 'error');
          }
        } catch { showToast('Revert failed', 'error'); }
        btn.textContent = 'Revert';
        btn.disabled = false;
      };
    });
  } catch { /* ignore */ }
}

function _initCheckpoints() {
  const toggle = document.getElementById('coder-checkpoints-toggle');
  const list = document.getElementById('coder-checkpoints-list');
  if (!toggle || !list) return;
  // _onEnterCoderMode can fire multiple times per page life (mode toggles,
  // workspace delete, terminal retry) and used to attach a fresh click
  // handler on each pass — N listeners flipped _checkpointsExpanded N
  // times per click, so even-N entries left the list stuck open.
  if (_checkpointsInitWired) return;
  _checkpointsInitWired = true;

  toggle.addEventListener('click', () => {
    _checkpointsExpanded = !_checkpointsExpanded;
    list.style.display = _checkpointsExpanded ? '' : 'none';
    toggle.classList.toggle('expanded', _checkpointsExpanded);
    if (_checkpointsExpanded) _loadCheckpoints();
  });

  // Manual save. Click must not bubble to the header toggle — otherwise
  // pressing Save would also collapse the list.
  const saveBtn = document.getElementById('coder-checkpoint-save-btn');
  saveBtn?.addEventListener('click', async (e) => {
    e.stopPropagation();
    if (!currentCoder().workspaceId) { showToast('No workspace selected', 'warning'); return; }
    const message = prompt('Checkpoint name:', 'Save point');
    if (message === null) return;
    const trimmed = message.trim();
    if (!trimmed) return;
    try {
      const resp = await fetch(
        `/api/coder/checkpoints/${encodeURIComponent(currentCoder().workspaceId)}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: trimmed }),
        },
      );
      const data = await resp.json();
      if (!resp.ok) {
        showToast(extractErrorMessage(data, 'Checkpoint failed'), 'error');
        return;
      }
      if (data.checkpoint) {
        showToast(`Checkpoint saved (${data.checkpoint})`, 'success');
      } else {
        showToast('Nothing changed since last checkpoint', 'info');
      }
      // Surface it in the list + tile count immediately.
      if (!_checkpointsExpanded) {
        _checkpointsExpanded = true;
        list.style.display = '';
        toggle.classList.add('expanded');
      }
      _loadCheckpoints();
    } catch (err) {
      showToast('Checkpoint failed: ' + err.message, 'error');
    }
  });

  // Start collapsed
  _checkpointsExpanded = false;
  list.style.display = 'none';
}

// ---------------------------------------------------------------------------
// Services panel — user-controlled lifecycle for workspace services that
// agent runs (or the user manually) have registered. Mirrors checkpoints
// structurally (collapsible drawer in the left panel) but persists across
// container restart since the rows live in coder_workspace_services + on-
// disk .augmentum/services.json.
// ---------------------------------------------------------------------------

let _servicesExpanded = false;
let _servicesInitWired = false;
let _servicesPollHandle = null;

function _serviceStatusDotColor(status) {
  switch ((status || '').toLowerCase()) {
    case 'running': return 'var(--success)';
    case 'error':   return 'var(--error)';
    default:        return 'var(--text-muted)';
  }
}

// Cache of duplicate service IDs per logical service, populated by the
// most recent _loadServices call. Used by the cleanup button to know
// which rows the backend should DELETE without re-walking the dedupe.
const _serviceDupeIds = new Map();

function _serviceDedupeKey(svc) {
  // Two distinct cohorts of phantom rows show up in practice and they
  // need different keys:
  //
  //  - Port-exposing services (Vite, web servers, APIs). The agent
  //    invokes the same logical service multiple ways across turns —
  //    ``npm run dev``, ``npx vite --host 0.0.0.0``, ``npm run dev --
  //    --host 0.0.0.0``. Same name, same port, same intent. Dropping
  //    command from the key collapses the variants into one entry.
  //
  //  - Outbound / portless services (cloudflared tunnel, localtunnel,
  //    file watchers). No ports to key on, and the command IS the
  //    intent — cf-tunnel vs localtunnel are genuinely different
  //    services even when they target the same upstream port.
  //    Keep command in the key.
  const name = (svc.name || '').trim();
  const ports = (svc.ports || []).slice().sort((a, b) => a - b).join(',');
  if (ports) {
    return ['p', name, ports].join('|');
  }
  return ['n', name, (svc.command || '').trim()].join('|');
}

function _serviceRank(status) {
  // Pick the most useful row when collapsing a group. Running wins so
  // the panel surfaces the live service. Among same-status rows, the
  // most recently updated wins so cleanup doesn't drop the live one.
  return ({ running: 3, paused: 2, stopped: 1 }[status] || 0);
}

function _dedupeServices(services) {
  const groups = new Map();
  for (const svc of services) {
    const key = _serviceDedupeKey(svc);
    const existing = groups.get(key);
    if (!existing) {
      groups.set(key, { winner: svc, dupes: [] });
      continue;
    }
    const winner = existing.winner;
    const swap = _serviceRank(svc.status) > _serviceRank(winner.status)
      || (_serviceRank(svc.status) === _serviceRank(winner.status)
          && (svc.updated_at || 0) > (winner.updated_at || 0));
    if (swap) {
      existing.dupes.push(winner.id);
      existing.winner = svc;
    } else {
      existing.dupes.push(svc.id);
    }
  }
  _serviceDupeIds.clear();
  const out = [];
  for (const g of groups.values()) {
    out.push({ ...g.winner, _dupeCount: g.dupes.length });
    if (g.dupes.length) _serviceDupeIds.set(g.winner.id, g.dupes);
  }
  return out;
}

async function _loadServices() {
  if (!currentCoder().workspaceId) return;
  const section = document.getElementById('coder-services');
  const listEl = document.getElementById('coder-services-list');
  const countEl = document.getElementById('coder-services-count');
  if (!section || !listEl) return;
  try {
    const resp = await fetch(
      `/api/coder/workspaces/${encodeURIComponent(currentCoder().workspaceId)}/services`,
    );
    if (!resp.ok) {
      section.hidden = true;
      return;
    }
    const data = await resp.json();
    const rawServices = Array.isArray(data?.services) ? data.services : [];
    const services = _dedupeServices(rawServices);
    // Surface the cleanup button when there's something to clean.
    const cleanupBtn = document.getElementById('coder-services-cleanup');
    if (cleanupBtn) {
      const dupeTotal = services.reduce((n, s) => n + (s._dupeCount || 0), 0);
      cleanupBtn.hidden = dupeTotal === 0;
      cleanupBtn.dataset.dupeTotal = String(dupeTotal);
      cleanupBtn.title = dupeTotal === 0
        ? ''
        : `Remove ${dupeTotal} duplicate row${dupeTotal === 1 ? '' : 's'} (folded into the visible entries)`;
    }
    // Hide the whole section when there's nothing to show — keeps the
    // left panel uncluttered for workspaces that never registered a
    // service. Re-shown the moment the first one lands.
    section.hidden = services.length === 0;
    if (services.length === 0) {
      if (countEl) countEl.textContent = '';
      listEl.innerHTML = '';
      return;
    }
    const running = services.filter(s => s.status === 'running').length;
    if (countEl) {
      countEl.textContent = running === services.length
        ? `(${services.length})`
        : `(${running}/${services.length})`;
    }
    listEl.innerHTML = services.map(svc => {
      const status = svc.status || 'unknown';
      const isRunning = status === 'running';
      const dotColor = _serviceStatusDotColor(status);
      const dupeBadge = (svc._dupeCount && svc._dupeCount > 0)
        ? `<span class="coder-svc-dupe-badge" title="${svc._dupeCount} duplicate row${svc._dupeCount === 1 ? '' : 's'} folded into this entry (older agent-restart phantoms)">+${svc._dupeCount}</span>`
        : '';
      // Port chips — clickable. Each links to the preview proxy if a
      // host port has been published for it, otherwise we fall back
      // to the in-container port label (still useful for copy-paste).
      const portChips = (svc.ports || []).map(p => {
        const url = `/api/coder/preview/${encodeURIComponent(currentCoder().workspaceId)}/${p}/`;
        return `<a class="coder-svc-port" href="${escapeHtml(url)}" target="_blank" rel="noopener" title="Open container port ${p} in browser">:${p}</a>`;
      }).join(' ');
      // Primary toggle — Start when stopped, Stop when running.
      const toggleBtn = isRunning
        ? `<button class="icon-btn small coder-svc-toggle" data-svc-action="stop" data-svc-id="${escapeHtml(svc.id)}" title="Stop service (config kept; restart with one click)">⏹</button>`
        : `<button class="icon-btn small coder-svc-toggle" data-svc-action="start" data-svc-id="${escapeHtml(svc.id)}" title="Start service with saved config">▶</button>`;
      // Truncate the command so long npm-script lines don't push the
      // action buttons off-screen on narrow panels.
      const cmd = (svc.command || '').length > 64
        ? svc.command.slice(0, 61) + '…'
        : svc.command;
      return `
        <div class="coder-svc-row" data-svc-id="${escapeHtml(svc.id)}">
          <div class="coder-svc-row-head">
            <span class="coder-svc-dot" style="background:${dotColor}" title="${escapeHtml(status)}"></span>
            <span class="coder-svc-name" title="${escapeHtml(svc.command || '')}">${escapeHtml(svc.name || svc.id)}</span>
            ${dupeBadge}
            ${portChips ? `<span class="coder-svc-ports">${portChips}</span>` : ''}
            ${toggleBtn}
            <button class="icon-btn small coder-svc-logs" data-svc-action="logs" data-svc-id="${escapeHtml(svc.id)}" title="View recent logs">📜</button>
            <button class="icon-btn small coder-svc-delete" data-svc-action="delete" data-svc-id="${escapeHtml(svc.id)}" data-svc-name="${escapeHtml(svc.name || svc.id)}" title="Remove this service entirely (config will not be remembered)">✕</button>
          </div>
          ${cmd ? `<div class="coder-svc-cmd"><code>${escapeHtml(cmd)}</code></div>` : ''}
        </div>`;
    }).join('');
  } catch (err) {
    console.debug('[Coder] services load failed', err);
    section.hidden = true;
  }
}

async function _handleServiceAction(action, svcId, btn, extra = {}) {
  if (!currentCoder().workspaceId || !svcId) return;
  const orig = btn.textContent;
  const setBusy = (label) => { btn.textContent = label; btn.disabled = true; };
  const restore = () => { btn.textContent = orig; btn.disabled = false; };
  try {
    if (action === 'start') {
      setBusy('…');
      const resp = await fetch(
        `/api/coder/workspaces/${encodeURIComponent(currentCoder().workspaceId)}/services/${encodeURIComponent(svcId)}/start`,
        { method: 'POST' },
      );
      if (!resp.ok) {
        let detail = '';
        try { detail = (await resp.json())?.error || ''; } catch (_) {}
        showToast(detail || 'Service start failed', 'error', 6000);
      }
      await _loadServices();
      return;
    }
    if (action === 'stop') {
      setBusy('…');
      const resp = await fetch(
        `/api/coder/workspaces/${encodeURIComponent(currentCoder().workspaceId)}/services/${encodeURIComponent(svcId)}/stop`,
        { method: 'POST' },
      );
      if (!resp.ok) {
        showToast('Service stop failed', 'error');
      }
      await _loadServices();
      return;
    }
    if (action === 'logs') {
      // Lightweight prompt-style logs viewer — opens a popup-style
      // alert so we don't need a full modal for v1. Future: inline
      // expandable panel.
      const resp = await fetch(
        `/api/coder/workspaces/${encodeURIComponent(currentCoder().workspaceId)}/services/${encodeURIComponent(svcId)}/logs?lines=200`,
      );
      if (!resp.ok) {
        showToast('Logs unavailable', 'error');
        return;
      }
      const data = await resp.json();
      const text = (data?.logs || '').trim() || '(no log output yet)';
      // Plain window.alert truncates aggressively in some browsers;
      // a temporary modal would be nicer but for v1 this matches the
      // checkpoint-save prompt style.
      window.alert(text.length > 4000 ? text.slice(text.length - 4000) : text);
      return;
    }
    if (action === 'delete') {
      const name = extra.name || svcId;
      if (!confirm(`Remove service "${name}"?\n\nThis stops the process AND forgets its configuration. To turn it off without losing the config, use the ⏹ button instead.`)) {
        return;
      }
      setBusy('…');
      const resp = await fetch(
        `/api/coder/workspaces/${encodeURIComponent(currentCoder().workspaceId)}/services/${encodeURIComponent(svcId)}`,
        { method: 'DELETE' },
      );
      if (!resp.ok) {
        showToast('Service removal failed', 'error');
      }
      await _loadServices();
      return;
    }
  } catch (err) {
    console.error('[Coder] service action failed', err);
    showToast(`${action} failed`, 'error');
    await _loadServices();
  } finally {
    restore();
  }
}

function _initServices() {
  // Re-entry guard mirroring _initCheckpoints — _onEnterCoderMode runs
  // multiple times per page life and we only want one click handler.
  const section = document.getElementById('coder-services');
  const toggle = document.getElementById('coder-services-toggle');
  const listEl = document.getElementById('coder-services-list');
  if (!section || !toggle || !listEl) return;
  if (_servicesInitWired) return;
  _servicesInitWired = true;

  // Start collapsed — mirrors checkpoints so the left panel doesn't
  // open noisily. The header still shows the running/total count chip
  // so the user knows there's something to expand without taking up
  // vertical space until they ask.
  _servicesExpanded = false;
  listEl.style.display = 'none';

  toggle.addEventListener('click', (e) => {
    // The cleanup button lives inside the toggle header; don't let its
    // click toggle the section collapsed.
    if (e.target.closest('#coder-services-cleanup')) return;
    _servicesExpanded = !_servicesExpanded;
    listEl.style.display = _servicesExpanded ? '' : 'none';
    toggle.classList.toggle('expanded', _servicesExpanded);
  });

  listEl.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-svc-action]');
    if (!btn) return;
    // Don't hijack port-chip anchor clicks — they're <a> elements,
    // not [data-svc-action] elements.
    const action = btn.dataset.svcAction;
    const svcId = btn.dataset.svcId;
    const extra = action === 'delete' ? { name: btn.dataset.svcName } : {};
    _handleServiceAction(action, svcId, btn, extra);
  });

  // Cleanup button: walks the duplicate-id map populated by the most
  // recent _loadServices call and DELETEs every folded row in one go.
  const cleanupBtn = document.getElementById('coder-services-cleanup');
  cleanupBtn?.addEventListener('click', async (e) => {
    e.stopPropagation();
    if (!currentCoder().workspaceId) return;
    const totalDupes = Number(cleanupBtn.dataset.dupeTotal || 0);
    if (totalDupes === 0) return;
    if (!confirm(`Remove ${totalDupes} duplicate service row${totalDupes === 1 ? '' : 's'}?\n\nThe visible service stays — only the older phantom copies (same name + command + ports) get cleaned up. The configs of the kept rows are untouched.`)) {
      return;
    }
    const originalLabel = cleanupBtn.textContent;
    cleanupBtn.textContent = '…';
    cleanupBtn.disabled = true;
    let removed = 0;
    let failed = 0;
    try {
      // Walk a copy of the map — _loadServices will rebuild it.
      for (const [, dupeIds] of [..._serviceDupeIds.entries()]) {
        for (const id of dupeIds) {
          try {
            const resp = await fetch(
              `/api/coder/workspaces/${encodeURIComponent(currentCoder().workspaceId)}/services/${encodeURIComponent(id)}`,
              { method: 'DELETE' },
            );
            if (resp.ok) {
              removed += 1;
            } else {
              failed += 1;
            }
          } catch (_) {
            failed += 1;
          }
        }
      }
      if (failed === 0) {
        showToast(`Removed ${removed} duplicate row${removed === 1 ? '' : 's'}`, 'success');
      } else {
        showToast(`Removed ${removed}; ${failed} failed`, 'warning', 6000);
      }
    } finally {
      cleanupBtn.textContent = originalLabel;
      cleanupBtn.disabled = false;
      await _loadServices();
    }
  });
}

function _startServicesPolling() {
  // Status drifts behind the UI when the user (or agent) runs commands
  // outside the panel: the process can die, port can change, etc.
  // 8s polling is cheap (one workspace exec per tick) and keeps the
  // dots honest. Clear on workspace switch.
  if (_servicesPollHandle) clearInterval(_servicesPollHandle);
  _servicesPollHandle = setInterval(() => {
    // Skip work while the tab is backgrounded (see workspace-status poll).
    if (document.hidden) return;
    const section = document.getElementById('coder-services');
    // No-op if section is hidden (zero services) so we don't poll for
    // workspaces that never had any.
    if (!section || section.hidden) return;
    _loadServices();
  }, 8000);
}

function _stopServicesPolling() {
  if (_servicesPollHandle) {
    clearInterval(_servicesPollHandle);
    _servicesPollHandle = null;
  }
}

// ---------------------------------------------------------------------------
// Git status, push/pull, settings
// ---------------------------------------------------------------------------

// Updates the header chip (branch label + dirty dot). Distinct from
// the file-tree decorator ``_refreshGitStatus(workspaceId)`` above,
// which paints per-file status badges from /git/file-status. Two
// endpoints, two caches, two consumers — same prefix was an
// accidental collision (SyntaxError under module strict mode).
// Cached so other handlers (commit panel, branch picker) can read
// branch / remote / log without a re-fetch. Cleared on workspace
// switch (see _stopGitPolling).
let _lastGitStatus = null;

async function _refreshGitHeaderStatus() {
  if (!currentCoder().workspaceId) return;
  try {
    const resp = await fetch(`/api/coder/workspaces/${currentCoder().workspaceId}/git/status`);
    if (!resp.ok) return;
    const data = await resp.json();
    _lastGitStatus = data;
    const controls = document.getElementById('coder-git-controls');
    const branchEl = document.getElementById('coder-git-branch');
    const dirtyEl = document.getElementById('coder-git-dirty');
    const aheadBehindEl = document.getElementById('coder-git-ahead-behind');
    if (!controls) return;
    controls.style.display = 'flex';
    const branch = data.branch || 'main';
    if (branchEl) {
      branchEl.textContent = branch;
      branchEl.title = data.remote
        ? `Branch ${branch} · tracking ${data.remote}`
        : `Branch ${branch} · no remote configured`;
    }
    if (dirtyEl) {
      dirtyEl.textContent = data.dirty ? '●' : '';
      dirtyEl.title = data.dirty ? 'Uncommitted changes' : 'Clean';
    }
    // Render ahead/behind counts as ↑N ↓N next to the branch chip.
    // Hidden when both zero — the chip stays tight on a clean clone.
    if (aheadBehindEl) {
      const ahead = Number(data.ahead || 0);
      const behind = Number(data.behind || 0);
      if (ahead === 0 && behind === 0) {
        aheadBehindEl.textContent = '';
        aheadBehindEl.title = '';
        aheadBehindEl.classList.add('hidden');
      } else {
        const parts = [];
        if (ahead > 0) parts.push(`↑${ahead}`);
        if (behind > 0) parts.push(`↓${behind}`);
        aheadBehindEl.textContent = parts.join(' ');
        const tipParts = [];
        if (ahead > 0) tipParts.push(`${ahead} local commit${ahead === 1 ? '' : 's'} not yet pushed`);
        if (behind > 0) tipParts.push(`${behind} remote commit${behind === 1 ? '' : 's'} not yet pulled`);
        aheadBehindEl.title = tipParts.join(' · ');
        aheadBehindEl.classList.remove('hidden');
      }
    }
  } catch { /* ignore */ }
}

function _startGitPolling() {
  _refreshGitHeaderStatus();
  _refreshPorts();
  if (currentCoder().gitPollInterval) clearInterval(currentCoder().gitPollInterval);
  if (currentCoder().portsPollInterval) clearInterval(currentCoder().portsPollInterval);
  // Gate on visibility at the timer level (not inside the fns) so on-demand
  // calls still work, but backgrounded tabs stop fetching + re-rendering.
  currentCoder().gitPollInterval = setInterval(() => {
    if (!document.hidden) _refreshGitHeaderStatus();
  }, 15000);
  // Ports change faster than branch state — a dev server that just
  // started should surface within a few seconds, not 15.
  currentCoder().portsPollInterval = setInterval(() => {
    if (!document.hidden) _refreshPorts();
  }, 5000);
}

function _stopGitPolling() {
  if (currentCoder().gitPollInterval) { clearInterval(currentCoder().gitPollInterval); currentCoder().gitPollInterval = null; }
  if (currentCoder().portsPollInterval) { clearInterval(currentCoder().portsPollInterval); currentCoder().portsPollInterval = null; }
  // Clear port badges when we leave — they'd be stale next time.
  const container = document.getElementById('coder-ports');
  if (container) {
    container.innerHTML = '';
    container.classList.add('hidden');
  }
  document.getElementById('coder-port-expose-btn')?.classList.add('hidden');
  document.getElementById('coder-preview-toggle-btn')?.classList.add('hidden');
}

async function _reconnectActiveTerminal() {
  if (!currentCoder().workspaceId || !currentCoder().dom.terminalPane) return;
  if (currentCoder().terminalId) {
    try { Terminal.destroy(currentCoder().terminalId); } catch {}
    currentCoder().terminalId = null;
  }
  currentCoder().dom.terminalPane.innerHTML = '';
  await Terminal.load();
  currentCoder().terminalId = await Terminal.create(currentCoder().dom.terminalPane, currentCoder().workspaceId);
  if (currentCoder().activeWorkbenchTab === 'terminal') {
    Terminal.focus(currentCoder().terminalId);
  }
}

async function _publishPortsForActiveWorkspace() {
  if (!currentCoder().workspaceId) {
    showToast('No workspace selected', 'warning');
    return;
  }
  const btn = document.getElementById('coder-port-expose-btn');
  if (!btn) return;
  if (!confirm(
    'Expose common dev-server ports for this workspace? This recreates the container against the same files and reconnects the terminal.'
  )) {
    return;
  }

  const oldText = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Exposing...';
  try {
    const resp = await fetch(
      `/api/coder/workspaces/${encodeURIComponent(currentCoder().workspaceId)}/ports/publish`,
      { method: 'POST' },
    );
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(extractErrorMessage(data, 'Failed to expose ports'));
    await _populateWorkspaceSelect();
    await _reconnectActiveTerminal();
    _startGitPolling();
    await _refreshPorts();
    showToast(
      data.changed ? 'Ports exposed for this workspace' : 'Ports were already exposed',
      'success',
    );
  } catch (err) {
    showToast(err.message || 'Failed to expose ports', 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = oldText;
  }
}

async function _refreshPorts() {
  const container = document.getElementById('coder-ports');
  const exposeBtn = document.getElementById('coder-port-expose-btn');
  const previewToggleBtn = document.getElementById('coder-preview-toggle-btn');
  if (!container || !currentCoder().workspaceId) return;
  try {
    const resp = await fetch(`/api/coder/workspaces/${encodeURIComponent(currentCoder().workspaceId)}/ports`);
    if (!resp.ok) return;
    const data = await resp.json();
    const ports = data.ports || [];
    const preview = data.preview || {};
    const state = preview.state || 'not_published';
    const published = state !== 'not_published';
    const listening = ports.filter(p => p.listening && p.host_port);
    currentCoder().previewPorts = ports;
    currentCoder().previewInfo = {
      state,
      published: !!preview.published,
      ready: !!preview.ready,
      ready_count: Number(preview.ready_count || listening.length || 0),
      primary_url: preview.primary_url || null,
      urls: Array.isArray(preview.urls) ? preview.urls : [],
    };
    if (exposeBtn) exposeBtn.classList.toggle('hidden', published);
    if (previewToggleBtn) previewToggleBtn.classList.toggle('hidden', state === 'not_published');
    if (listening.length === 0) {
      if (state === 'published_idle') {
        const lanNote = currentCoder().lanAccessible ? ' (LAN-reachable)' : '';
        // Single-line span content: the hint ellipsizes via CSS
        // (white-space: nowrap), so template-literal newlines/indent
        // would render as leading whitespace inside the pill.
        container.innerHTML = `<span class="coder-port-hint" title="Ports are exposed for this workspace${lanNote}. Start a dev server on a common port like 3000, 5173, 8000, or 8080 to get a clickable preview badge.">Ports exposed${lanNote} • waiting for a dev server</span>`;
        container.classList.remove('hidden');
      } else {
        container.innerHTML = '';
        container.classList.add('hidden');
      }
      _renderPreviewPane();
      return;
    }
    // Render one badge per listening port. Clicking opens the in-app
    // preview surface for that port; the preview pane itself exposes
    // the external-open action when users want a separate tab.
    const lanHost = currentCoder().lanAccessible ? location.hostname : '';
    const gateUrls = data.gate_urls || [];
    const gateUrl = gateUrls[0] || '';
    container.innerHTML = listening.map(p => {
      const lanUrl = lanHost && p.host_port ? `http://${lanHost}:${p.host_port}` : '';
      let title = lanUrl
        ? `Container :${p.container_port} → host :${p.host_port} • LAN: ${lanUrl}`
        : `Container port ${p.container_port} → host ${p.host_port}`;
      if (gateUrl) title += ` • HTTPS: ${gateUrl}`;
      return `
      <button class="coder-port-badge"
         type="button"
         data-preview-url="/api/coder/preview/${encodeURIComponent(currentCoder().workspaceId)}/${p.container_port}/"
         ${lanUrl ? `data-lan-url="${escapeHtml(lanUrl)}"` : ''}
         title="${escapeHtml(title)}">
        <span class="coder-port-badge-dot" ${currentCoder().lanAccessible ? 'style="background:var(--accent,#6cf)"' : ''}></span>
        :${p.container_port}${currentCoder().lanAccessible ? ' <span style="font-size:9px;opacity:.7">LAN</span>' : ''}
      </button>`;
    }).join('');
    if (gateUrl && currentCoder().lanAccessible) {
      container.innerHTML += `<a href="${escapeHtml(gateUrl)}" target="_blank" rel="noopener"
        class="coder-port-hint" style="font-size:10px;text-decoration:none;color:var(--accent,#6cf)"
        title="HTTPS gate URL — reachable from your network with Augmentum auth. For external access: Tailscale, port forwarding, or localtunnel/Cloudflare Tunnel."
      >${escapeHtml(gateUrl.replace(/^https?:\/\//, ''))}</a>`;
    }
    container.querySelectorAll('[data-preview-url]').forEach((btn) => {
      btn.addEventListener('click', () => _openPreview(btn.dataset.previewUrl || ''));
    });
    container.classList.remove('hidden');
    _renderPreviewPane();
  } catch { /* ignore transient polling failures */ }
}

async function _openNewWorkspaceModal() {
  // Refresh the tooling-profile cache before building the modal so the
  // dropdown reflects server truth (catalog grows in profiles.py).
  await _loadToolingProfiles().catch(() => {});
  let modal = document.getElementById('coder-new-workspace-modal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'coder-new-workspace-modal';
    modal.className = 'modal-overlay hidden';
    modal.innerHTML = `
      <div class="modal" style="width:min(460px,95vw)">
        <div class="modal-header">
          <span class="modal-title">New Workspace</span>
          <button class="icon-btn small" id="coder-nw-close">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>
        <div class="modal-body" style="display:flex;flex-direction:column;gap:var(--space-sm)">
          <div class="field-group">
            <label class="field-label">Workspace Type</label>
            <select class="field-select" id="coder-nw-kind">
              <option value="regular">Regular — terminal, files, AI coding agent</option>
              <option value="bug_finder">Bug Finder — autonomous audit runs</option>
            </select>
            <div class="coder-nw-kind-hint" id="coder-nw-kind-hint" style="font-size:11px;color:var(--text-muted);margin-top:4px;line-height:1.4">
              A Bug Finder workspace adds an audit surface to the workbench. The type is fixed once created — switch by picking a different workspace.
            </div>
          </div>
          <div class="field-group">
            <label class="field-label">Workspace Name</label>
            <input type="text" class="field-input" id="coder-nw-name" placeholder="my-project" value="workspace">
          </div>
          <div class="field-group">
            <label class="field-label">Clone from Repository <span style="color:var(--text-muted);font-weight:400">(optional)</span></label>
            <input type="text" class="field-input" id="coder-nw-url" placeholder="https://github.com/user/repo.git">
          </div>
          <div class="field-group">
            <label class="field-label">Branch <span style="color:var(--text-muted);font-weight:400">(optional)</span></label>
            <input type="text" class="field-input" id="coder-nw-branch" placeholder="main">
          </div>
          <div class="field-group">
            <label class="field-label">Tooling Profile</label>
            <select class="field-select" id="coder-nw-tooling-profile">
              ${_toolingProfileOptions('browser')}
            </select>
          </div>
          <label class="coder-publish-ports-row" title="Publish common dev-server ports (3000, 5173, 8000, etc.) to 127.0.0.1 so dev servers running inside the workspace are reachable from your browser.">
            <input type="checkbox" id="coder-nw-publish-ports">
            <span>Expose dev-server ports (3000, 5173, 8000, …)</span>
          </label>
          <button class="btn btn-primary btn-full" id="coder-nw-create">Create</button>
          <button class="btn btn-ghost btn-full" id="coder-nw-self-test" title="Seed a workspace with the live host augmentum source (tracked + untracked-not-ignored). Requires the /host-augmentum-src bind mount in compose.yaml.">
            Seed from Augmentum source (self-test)
          </button>
        </div>
      </div>`;
    document.body.appendChild(modal);

    modal.querySelector('#coder-nw-close').addEventListener('click', () => modal.classList.add('hidden'));
    modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.add('hidden'); });

    modal.querySelector('#coder-nw-create').addEventListener('click', async () => {
      const name = modal.querySelector('#coder-nw-name').value.trim() || 'workspace';
      const url = modal.querySelector('#coder-nw-url').value.trim() || null;
      const branch = modal.querySelector('#coder-nw-branch').value.trim() || null;
      const toolingProfile = modal.querySelector('#coder-nw-tooling-profile')?.value || 'browser';
      const kind = modal.querySelector('#coder-nw-kind')?.value || 'regular';

      // Auto-fill name from repo URL if user left it as default
      const finalName = (name === 'workspace' && url)
        ? url.split('/').pop()?.replace('.git', '') || name
        : name;

      const btn = modal.querySelector('#coder-nw-create');
      btn.textContent = url ? 'Cloning...' : 'Creating...';
      btn.disabled = true;

      const options = {};
      if (url) options.git_url = url;
      if (branch) options.git_branch = branch;
      options.tooling_profile = toolingProfile;
      options.kind = kind;
      if (modal.querySelector('#coder-nw-publish-ports')?.checked) {
        options.publish_ports = true;
      }

      const result = await createWorkspace(finalName, options);
      btn.textContent = 'Create';
      btn.disabled = false;

      if (result?.id) {
        modal.classList.add('hidden');
        _populateWorkspaceSelect();
        // Reset form for next use
        modal.querySelector('#coder-nw-name').value = 'workspace';
        modal.querySelector('#coder-nw-url').value = '';
        modal.querySelector('#coder-nw-branch').value = '';
        modal.querySelector('#coder-nw-tooling-profile').value = 'browser';
        modal.querySelector('#coder-nw-kind').value = 'regular';
      } else {
        showToast('Failed to create workspace', 'error');
      }
    });

    modal.querySelector('#coder-nw-self-test').addEventListener('click', async () => {
      const btn = modal.querySelector('#coder-nw-self-test');
      const label = btn.textContent;
      btn.textContent = 'Seeding...';
      btn.disabled = true;
      try {
        const resp = await fetch('/api/coder/workspaces/self-test', { method: 'POST' });
        const data = await resp.json();
        if (!resp.ok) {
          showToast(data?.error || 'Self-test workspace failed', 'error', 6000);
          return;
        }
        modal.classList.add('hidden');
        currentCoder().chatHistory = [];
        currentCoder().conversation?.clear();
        currentCoder().missionPanel?.clear();
        currentCoder().workspaceId = data.id;
        _persistActiveWorkspaceId(data.id);
        currentCoder().conversation?.setWorkspaceId(data.id);
        showToast(`Seeded with ${data.seeded_files} files`, 'success', 4000);
        _populateWorkspaceSelect();
        // Re-route the surface to the freshly created workspace so the
        // file tree + terminal bind to it immediately.
        document.dispatchEvent(new CustomEvent('coder-workspace-changed', {
          detail: { workspaceId: data.id },
        }));
      } catch (err) {
        console.error('self-test workspace failed', err);
        showToast('Self-test workspace failed', 'error');
      } finally {
        btn.textContent = label;
        btn.disabled = false;
      }
    });
  }
  modal.classList.remove('hidden');
  setTimeout(() => modal.querySelector('#coder-nw-name')?.focus(), 100);
}

// ---------------------------------------------------------------------------
// Workspace Manager modal — list every workspace with per-row actions
// ---------------------------------------------------------------------------

function _formatRelativeTime(epochSeconds) {
  if (!epochSeconds) return 'never';
  const delta = Date.now() / 1000 - epochSeconds;
  if (delta < 60) return 'just now';
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  if (delta < 86400 * 30) return `${Math.floor(delta / 86400)}d ago`;
  if (delta < 86400 * 365) return `${Math.floor(delta / (86400 * 30))}mo ago`;
  return `${Math.floor(delta / (86400 * 365))}y ago`;
}

function _statusDot(status) {
  // Returns inline-SVG markup for a colored status indicator. Inline
  // because the manager modal is a one-off — not worth adding to the
  // shared icon set.
  const color = {
    running: 'var(--success)',
    paused: 'var(--warning)',
    stopped: 'var(--text-muted)',
  }[status] || 'var(--text-muted)';
  return `<span class="coder-wm-dot" style="background:${color}" title="${escapeHtml(status)}"></span>`;
}

function _formatAbsoluteDate(epochSeconds) {
  if (!epochSeconds) return 'unknown';
  const d = new Date(epochSeconds * 1000);
  return d.toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

function _renderWorkspaceDetails(w) {
  // Snapshot of everything we track for a workspace, formatted for
  // human reading. Mirrors the ContainerInfo dataclass fields so a
  // reader can map UI ↔ schema. Lazy-fetched stats (checkpoint count,
  // etc.) get patched in after expand via _hydrateWorkspaceDetails.
  const row = (label, value, opts = {}) => {
    if (value === null || value === undefined || value === '') {
      if (!opts.includeEmpty) return '';
      value = '—';
    }
    const mono = opts.mono ? ' coder-wm-detail-mono' : '';
    return `<div class="coder-wm-detail-row">
      <span class="coder-wm-detail-label">${escapeHtml(label)}</span>
      <span class="coder-wm-detail-value${mono}">${escapeHtml(String(value))}</span>
    </div>`;
  };
  // Friendlier labels for the kind/planning_mode strings.
  const kindLabel = ({
    regular: 'Regular',
    bug_finder: 'Bug Finder',
  }[w.kind] || w.kind || 'Regular');
  const planningLabel = ({
    auto: 'Auto (model runs freely)',
    default: 'Approve (per-tool prompts)',
    plan: 'Plan (outline before edits)',
  }[w.planning_mode] || w.planning_mode || 'Auto');
  const safeguardsLabel = w.safeguards_enabled === false
    ? 'Off (model bypasses soft circuit breakers)'
    : 'On (soft circuit breakers active)';
  const alwaysOnLabel = w.always_on
    ? 'On (exempt from idle reaper)'
    : 'Off (auto-stops after idle timeout)';
  const resources = `${w.resources_cpu || '?'} CPU · ${w.resources_memory || '?'} RAM`;
  return `
    <div class="coder-wm-details-grid">
      ${row('Kind', kindLabel)}
      ${row('Tooling profile', w.tooling_profile || 'browser')}
      ${row('Resources', resources)}
      ${row('Created', _formatAbsoluteDate(w.created_at))}
      ${row('Last active', _formatAbsoluteDate(w.last_active))}
      ${row('Lifecycle', alwaysOnLabel)}
      ${row('Planning mode', planningLabel)}
      ${row('Safeguards', safeguardsLabel)}
      ${row('Workspace ID', w.id, { mono: true })}
      ${row('Container ID', w.container_id ? w.container_id.slice(0, 12) : '(no container — recreate to start)', { mono: true })}
      ${w.kind === 'bug_finder' ? row('Verifier model', w.bug_finder_verifier_model || '(self-verify)') : ''}
      ${row('Git URL', w.git_url, { mono: true })}
      ${row('Project ID', w.project_id, { mono: true })}
      <div class="coder-wm-detail-row" data-wm-stat="checkpoints">
        <span class="coder-wm-detail-label">Checkpoints</span>
        <span class="coder-wm-detail-value coder-wm-detail-loading">loading…</span>
      </div>
    </div>`;
}

async function _hydrateWorkspaceDetails(wsId, detailsEl) {
  // Lazy stats: only fetched when the user actually opens Details, so
  // the modal first-render cost is bounded by the workspace list alone.
  const slot = detailsEl.querySelector('[data-wm-stat="checkpoints"] .coder-wm-detail-value');
  if (!slot) return;
  // ``list_checkpoints`` returns a bounded array but no honest count,
  // so we ask for a generous cap and signal overflow rather than
  // claiming an exact number we don't know.
  const CAP = 200;
  try {
    const resp = await fetch(`/api/coder/checkpoints/${encodeURIComponent(wsId)}?limit=${CAP}`);
    if (!resp.ok) {
      slot.classList.remove('coder-wm-detail-loading');
      slot.textContent = '—';
      return;
    }
    const data = await resp.json();
    const entries = Array.isArray(data?.checkpoints) ? data.checkpoints : [];
    slot.classList.remove('coder-wm-detail-loading');
    if (entries.length === 0) {
      slot.textContent = 'None yet';
    } else if (entries.length >= CAP) {
      slot.textContent = `${CAP}+`;
    } else {
      slot.textContent = String(entries.length);
    }
  } catch {
    slot.classList.remove('coder-wm-detail-loading');
    slot.textContent = '—';
  }
}

async function _renderWorkspaceManagerList(listEl, filter = null) {
  // ``filter=null`` defaults to "whatever the search input currently
  // holds" so post-action re-renders preserve the user's filter
  // without every action-handler call site threading it through.
  if (filter === null) {
    filter = document.getElementById('coder-wm-search')?.value || '';
  }
  if (_wmArchivedView) {
    await _renderArchivedManagerList(listEl);
    return;
  }
  // Pull fresh workspaces straight from the server every render. The
  // server-side cache (see ContainerManager._docker_state_map) means
  // repeated re-renders after actions are nearly free in IPC terms.
  const all = await _fetchWorkspaces();
  // Client-side filter — server returns the user's full list and we
  // narrow on name OR truncated id. Empty filter shows everything.
  const needle = (filter || '').trim().toLowerCase();
  const workspaces = needle
    ? all.filter(w =>
        (w.name || '').toLowerCase().includes(needle)
        || w.id.toLowerCase().includes(needle))
    : all;
  if (workspaces.length === 0) {
    listEl.innerHTML = needle
      ? `<div class="coder-wm-empty">
          <p>No workspaces match "${escapeHtml(needle)}".</p>
          <p style="color:var(--text-muted);font-size:var(--text-xs)">Clear the search to see all ${all.length}.</p>
        </div>`
      : `<div class="coder-wm-empty">
          <p>No workspaces yet.</p>
          <p style="color:var(--text-muted);font-size:var(--text-xs)">Use the + button in the workspace bar to create one.</p>
        </div>`;
    return;
  }
  // Sort: running first, then paused, then stopped — and within each
  // group, most-recently-active first. Matches what the user usually
  // wants to see at the top.
  const order = { running: 0, paused: 1, stopped: 2 };
  workspaces.sort((a, b) => {
    const aOrd = order[a.status] ?? 3;
    const bOrd = order[b.status] ?? 3;
    if (aOrd !== bOrd) return aOrd - bOrd;
    return (b.last_active || 0) - (a.last_active || 0);
  });
  listEl.innerHTML = workspaces.map(w => {
    const status = w.status || 'stopped';
    const isRunning = status === 'running';
    const isPaused = status === 'paused';
    const isStopped = status === 'stopped';
    const isActive = w.id === currentCoder().workspaceId;
    // Lifecycle slot — primary button changes by state:
    //   stopped/paused → Start/Resume (the /start route handles both)
    //   running        → Pause + Stop (pause is the soft option,
    //                    stop is the hard one)
    const lifecycleBtns = (isStopped || isPaused)
      ? `<button class="btn btn-sm" data-wm-action="start" data-wm-id="${escapeHtml(w.id)}" title="${isPaused ? 'Resume from cgroup freeze' : 'Start container'}">${isPaused ? '▶ Resume' : '▶ Start'}</button>`
      : `<button class="btn btn-sm" data-wm-action="pause" data-wm-id="${escapeHtml(w.id)}" title="Freeze the container (cgroup pause). RAM stays held; sub-second resume.">⏸ Pause</button>
         <button class="btn btn-sm" data-wm-action="stop" data-wm-id="${escapeHtml(w.id)}" title="Stop container — RAM freed, restart takes longer">⏹ Stop</button>`;
    const alwaysOnChip = w.always_on
      ? `<span class="coder-wm-chip coder-wm-chip-always-on" title="Exempt from idle reaper">always-on</span>`
      : '';
    const kindChip = (w.kind && w.kind !== 'regular')
      ? `<span class="coder-wm-chip">${escapeHtml(w.kind)}</span>`
      : '';
    const activeChip = isActive
      ? `<span class="coder-wm-chip coder-wm-chip-active">active</span>`
      : '';
    return `
      <div class="coder-wm-row" data-wm-id="${escapeHtml(w.id)}">
        <div class="coder-wm-row-head">
          ${_statusDot(status)}
          <span class="coder-wm-name">${escapeHtml(w.name || w.id)}</span>
          ${activeChip}${kindChip}${alwaysOnChip}
        </div>
        <div class="coder-wm-row-meta">
          <code>${escapeHtml(w.id.slice(0, 12))}</code>
          <span class="coder-wm-sep">·</span>
          <span>${isRunning ? 'Active' : 'Last active'} ${_formatRelativeTime(w.last_active)}</span>
        </div>
        <div class="coder-wm-row-actions">
          <button class="btn btn-sm btn-primary" data-wm-action="open" data-wm-id="${escapeHtml(w.id)}" ${isActive ? 'disabled' : ''}>
            ${isActive ? 'Current' : 'Open'}
          </button>
          ${lifecycleBtns}
          <button class="btn btn-sm btn-ghost" data-wm-action="toggle-always-on" data-wm-id="${escapeHtml(w.id)}" data-wm-current="${w.always_on ? '1' : '0'}" title="Toggle always-on (exempt from idle reaper)">
            ${w.always_on ? '☼ on-demand' : '☼ always-on'}
          </button>
          <button class="btn btn-sm btn-ghost" data-wm-action="rename" data-wm-id="${escapeHtml(w.id)}" data-wm-name="${escapeHtml(w.name || '')}" title="Rename workspace">✎ Rename</button>
          <button class="btn btn-sm btn-ghost" data-wm-action="export" data-wm-id="${escapeHtml(w.id)}" title="Download workspace as .tar.gz (excludes node_modules / .venv / target / etc.)">⬇ Export</button>
          <button class="btn btn-sm btn-ghost" data-wm-action="toggle-details" data-wm-id="${escapeHtml(w.id)}" title="Show what we track for this workspace" aria-expanded="false">▸ Details</button>
          <button class="btn btn-sm btn-danger" data-wm-action="delete" data-wm-id="${escapeHtml(w.id)}" data-wm-name="${escapeHtml(w.name || w.id)}" title="Archive (keep files, restorable) — or completely remove">🗑 Delete</button>
        </div>
        <div class="coder-wm-details" data-wm-details-id="${escapeHtml(w.id)}" hidden>
          ${_renderWorkspaceDetails(w)}
        </div>
      </div>`;
  }).join('');
}

function _formatBytes(n) {
  n = Number(n) || 0;
  if (n <= 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n >= 100 || i === 0 ? Math.round(n) : n.toFixed(1)} ${units[i]}`;
}

// Delete confirmation with an "also delete files" opt-in. Native confirm()
// can't hold a checkbox, so this is a tiny promise-wrapped modal. Resolves to
// null (cancelled) or { purge } where purge=true means "completely remove the
// on-disk volume too" (irreversible). Default is archive (purge=false).
function _confirmWorkspaceRemoval(name) {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.innerHTML = `
      <div class="modal" style="width:min(460px,95vw)">
        <div class="modal-header">
          <span class="modal-title">Remove workspace</span>
        </div>
        <div class="modal-body" style="display:flex;flex-direction:column;gap:var(--space-sm)">
          <p style="margin:0">Archive <strong>${escapeHtml(name)}</strong>?</p>
          <p style="margin:0;color:var(--text-muted);font-size:var(--text-xs)">
            Archiving removes the container to reclaim its space but keeps the
            files and task history on disk — you can restore it natively any time
            from the Archived tab.
          </p>
          <label style="display:flex;align-items:flex-start;gap:var(--space-xs);margin-top:var(--space-xs);cursor:pointer">
            <input type="checkbox" id="coder-rm-purge" style="margin-top:3px">
            <span style="font-size:var(--text-sm)">Completely remove — also delete all files
              <span style="color:var(--danger,#e5534b))">(cannot be undone)</span>
            </span>
          </label>
        </div>
        <div class="modal-footer" style="display:flex;justify-content:flex-end;gap:var(--space-xs)">
          <button class="btn btn-sm btn-ghost" id="coder-rm-cancel">Cancel</button>
          <button class="btn btn-sm btn-danger" id="coder-rm-confirm">Archive</button>
        </div>
      </div>`;
    document.body.appendChild(overlay);
    const purgeBox = overlay.querySelector('#coder-rm-purge');
    const confirmBtn = overlay.querySelector('#coder-rm-confirm');
    // Reflect the destructive choice in the button label so the user sees
    // exactly what the primary action will do.
    purgeBox.addEventListener('change', () => {
      confirmBtn.textContent = purgeBox.checked ? 'Delete everything' : 'Archive';
    });
    const close = (result) => { overlay.remove(); resolve(result); };
    overlay.querySelector('#coder-rm-cancel').addEventListener('click', () => close(null));
    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(null); });
    confirmBtn.addEventListener('click', () => close({ purge: purgeBox.checked }));
  });
}

// Archived-workspaces view inside the manager modal. Cards show name, size
// reclaimed, when archived, and accumulated task progression, with Restore /
// Export / Delete actions.
async function _renderArchivedManagerList(listEl) {
  let archives = [];
  try {
    const resp = await fetch('/api/coder/archives');
    if (resp.ok) archives = (await resp.json())?.archives || [];
  } catch (_) { /* render empty below */ }
  if (archives.length === 0) {
    listEl.innerHTML = `<div class="coder-wm-empty">
      <p>No archived workspaces.</p>
      <p style="color:var(--text-muted);font-size:var(--text-xs)">Deleting a workspace archives it here — the container is removed but files and task history are kept, restorable any time.</p>
    </div>`;
    return;
  }
  const totalBytes = archives.reduce((s, a) => s + (Number(a.size_bytes) || 0), 0);
  const header = `<div class="coder-wm-arch-summary" style="padding:var(--space-xs) var(--space-sm);color:var(--text-muted);font-size:var(--text-xs)">
    ${archives.length} archived · ${_formatBytes(totalBytes)} on disk (restorable, or reclaim by deleting)
  </div>`;
  listEl.innerHTML = header + archives.map(a => {
    const t = a.tasks || { total: 0, done: 0, items: [] };
    const taskChip = t.total > 0
      ? `<span class="coder-wm-chip" title="Task progression captured before archiving">${t.done}/${t.total} tasks</span>`
      : '';
    const gone = !a.volume_present;
    const goneChip = gone
      ? `<span class="coder-wm-chip" style="color:var(--danger,#e5534b)" title="The data volume is gone — only the record remains">volume missing</span>`
      : '';
    const taskList = (t.items && t.items.length)
      ? `<ul class="coder-wm-arch-tasks" style="margin:var(--space-xs) 0 0;padding-left:var(--space-md);max-height:120px;overflow:auto">
          ${t.items.map(it => {
            const done = String(it.status || '').toLowerCase();
            const mark = ['done', 'complete', 'completed', 'verified'].includes(done) ? '✓' : '·';
            return `<li style="font-size:var(--text-xs);color:var(--text-muted)">${mark} ${escapeHtml(it.text || '')}</li>`;
          }).join('')}
        </ul>`
      : '';
    return `
      <div class="coder-wm-row" data-wm-id="${escapeHtml(a.id)}">
        <div class="coder-wm-row-head">
          <span class="coder-wm-name">${escapeHtml(a.name || a.id)}</span>
          <span class="coder-wm-chip">${_formatBytes(a.size_bytes)}</span>
          ${taskChip}${goneChip}
        </div>
        <div class="coder-wm-row-meta">
          <code>${escapeHtml(a.id.slice(0, 12))}</code>
          <span class="coder-wm-sep">·</span>
          <span>archived ${_formatRelativeTime(a.archived_at)}</span>
        </div>
        <div class="coder-wm-row-actions">
          <button class="btn btn-sm btn-primary" data-wm-action="restore" data-wm-id="${escapeHtml(a.id)}" ${gone ? 'disabled title="Data volume is gone"' : 'title="Respawn a container onto the saved files"'}>↻ Restore</button>
          <button class="btn btn-sm btn-ghost" data-wm-action="export" data-wm-id="${escapeHtml(a.id)}" ${gone ? 'disabled' : ''} title="Download as .tar.gz">⬇ Export</button>
          <button class="btn btn-sm btn-danger" data-wm-action="purge" data-wm-id="${escapeHtml(a.id)}" data-wm-name="${escapeHtml(a.name || a.id)}" title="Completely remove — delete the files too (cannot be undone)">🗑 Delete</button>
        </div>
        ${taskList}
      </div>`;
  }).join('');
}

async function _handleWorkspaceManagerAction(action, wsId, btn, listEl, extra = {}) {
  // Centralised action handler — every row button routes through here
  // so post-action refresh is one place instead of N copy-pasted blocks.
  const setBusy = (label) => { btn.textContent = label; btn.disabled = true; };
  const ok = async (msg) => {
    showToast(msg, 'success');
    await _renderWorkspaceManagerList(listEl);
  };
  try {
    if (action === 'open') {
      // Close the modal first so the workspace switch has a clean stage.
      document.getElementById('coder-workspace-manager-modal')?.classList.add('hidden');
      await openWorkspaceById(wsId);
      return;
    }
    if (action === 'start') {
      setBusy('Starting…');
      const resp = await fetch(`/api/coder/workspaces/${encodeURIComponent(wsId)}/start`, { method: 'POST' });
      if (!resp.ok) {
        let detail = '';
        try { detail = (await resp.json())?.error || ''; } catch (_) {}
        showToast(detail || 'Start failed', 'error', 6000);
        await _renderWorkspaceManagerList(listEl);
        return;
      }
      await ok('Workspace started');
      return;
    }
    if (action === 'stop') {
      setBusy('Stopping…');
      const resp = await fetch(`/api/coder/workspaces/${encodeURIComponent(wsId)}/stop`, { method: 'POST' });
      if (!resp.ok) {
        showToast('Stop failed', 'error');
        await _renderWorkspaceManagerList(listEl);
        return;
      }
      await ok('Workspace stopped');
      return;
    }
    if (action === 'pause') {
      setBusy('Pausing…');
      const resp = await fetch(`/api/coder/workspaces/${encodeURIComponent(wsId)}/pause`, { method: 'POST' });
      if (!resp.ok) {
        showToast('Pause failed', 'error');
        await _renderWorkspaceManagerList(listEl);
        return;
      }
      await ok('Workspace paused (RAM held, sub-second resume)');
      return;
    }
    if (action === 'rename') {
      // Use the existing name as the prompt seed so it's editable
      // rather than starting blank. Returning the same value or
      // empty cancels.
      const current = extra.name || '';
      const next = window.prompt('Rename workspace:', current);
      if (next === null) return;
      const trimmed = next.trim();
      if (!trimmed || trimmed === current) return;
      if (trimmed.length > 80) {
        showToast('Name too long (max 80 characters)', 'warning');
        return;
      }
      setBusy('Renaming…');
      const resp = await fetch(`/api/coder/workspaces/${encodeURIComponent(wsId)}/name`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: trimmed }),
      });
      if (!resp.ok) {
        let detail = '';
        try { detail = (await resp.json())?.error || ''; } catch (_) {}
        showToast(detail || 'Rename failed', 'error');
        await _renderWorkspaceManagerList(listEl);
        return;
      }
      // Refresh the workspace bar too if this was the active workspace
      // — the dropdown displays the name.
      if (wsId === currentCoder().workspaceId) {
        if (currentCoder().dom.filesTitle) currentCoder().dom.filesTitle.textContent = trimmed;
        _populateWorkspaceSelect();
      }
      await ok('Renamed');
      return;
    }
    if (action === 'export') {
      // Existing GET endpoint streams a .tar.gz with the right
      // Content-Disposition; navigating to it triggers a browser
      // download without leaving the page. No button-busy state since
      // the request completes immediately on the server side and the
      // stream lives in the browser's download UI.
      window.location.href = `/api/coder/workspaces/${encodeURIComponent(wsId)}/export`;
      return;
    }
    if (action === 'toggle-details') {
      const detailsEl = listEl.querySelector(`[data-wm-details-id="${CSS.escape(wsId)}"]`);
      if (!detailsEl) return;
      const wasHidden = detailsEl.hasAttribute('hidden');
      if (wasHidden) {
        detailsEl.removeAttribute('hidden');
        btn.setAttribute('aria-expanded', 'true');
        btn.textContent = '▾ Details';
        // Lazy-hydrate the derived stats. Idempotent — re-running just
        // re-fetches the same numbers, no churn.
        await _hydrateWorkspaceDetails(wsId, detailsEl);
      } else {
        detailsEl.setAttribute('hidden', '');
        btn.setAttribute('aria-expanded', 'false');
        btn.textContent = '▸ Details';
      }
      return;
    }
    if (action === 'toggle-always-on') {
      const next = extra.current !== '1';
      setBusy(next ? 'Pinning…' : 'Releasing…');
      const resp = await fetch(`/api/coder/workspaces/${encodeURIComponent(wsId)}/always-on`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ always_on: next }),
      });
      if (!resp.ok) {
        showToast('Toggle failed', 'error');
        await _renderWorkspaceManagerList(listEl);
        return;
      }
      await ok(next ? 'Always-on enabled' : 'Always-on disabled');
      return;
    }
    if (action === 'delete') {
      // Archive by default; the checkbox opts into a full purge (volume too).
      const name = extra.name || wsId.slice(0, 12);
      const choice = await _confirmWorkspaceRemoval(name);
      if (!choice) return;
      const purge = choice.purge;
      setBusy(purge ? 'Removing…' : 'Archiving…');
      const url = purge
        ? `/api/coder/workspaces/${encodeURIComponent(wsId)}?purge=1`
        : `/api/coder/workspaces/${encodeURIComponent(wsId)}`;
      const resp = await fetch(url, { method: 'DELETE' });
      if (!resp.ok) {
        showToast(purge ? 'Remove failed' : 'Archive failed', 'error');
        await _renderWorkspaceManagerList(listEl);
        return;
      }
      // If this was the active workspace, clear the local active pointer so
      // the next mode-entry doesn't try to bind to it.
      if (wsId === currentCoder().workspaceId) {
        currentCoder().chatHistory = [];
        currentCoder().conversation?.clear();
        currentCoder().missionPanel?.clear();
        currentCoder().workspaceId = null;
        _persistActiveWorkspaceId('');
      }
      await ok(purge ? 'Workspace removed' : 'Workspace archived');
      // Refresh the workspace-bar dropdown too (manager modal isn't the
      // only place showing the list).
      _populateWorkspaceSelect();
      return;
    }
    if (action === 'restore') {
      setBusy('Restoring…');
      const resp = await fetch(`/api/coder/archives/${encodeURIComponent(wsId)}/restore`, { method: 'POST' });
      if (!resp.ok) {
        let detail = '';
        try { detail = (await resp.json())?.error || ''; } catch (_) {}
        showToast(detail || 'Restore failed', 'error', 6000);
        await _renderWorkspaceManagerList(listEl);
        return;
      }
      await ok('Workspace restored');
      _populateWorkspaceSelect();
      return;
    }
    if (action === 'purge') {
      const name = extra.name || wsId.slice(0, 12);
      if (!confirm(`Permanently delete "${name}" and all its files?\n\nThis frees the disk space but cannot be undone — the volume is destroyed.`)) {
        return;
      }
      setBusy('Deleting…');
      const resp = await fetch(`/api/coder/archives/${encodeURIComponent(wsId)}`, { method: 'DELETE' });
      if (!resp.ok) {
        showToast('Delete failed', 'error');
        await _renderWorkspaceManagerList(listEl);
        return;
      }
      await ok('Workspace deleted');
      _populateWorkspaceSelect();
      return;
    }
  } catch (err) {
    console.error('[Coder] Workspace manager action failed', err);
    showToast(`${action} failed: ${err.message || 'unknown error'}`, 'error');
    await _renderWorkspaceManagerList(listEl);
  }
}

let _wmArchivedView = false;

async function _openWorkspaceManagerModal() {
  let modal = document.getElementById('coder-workspace-manager-modal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'coder-workspace-manager-modal';
    modal.className = 'modal-overlay hidden';
    modal.innerHTML = `
      <div class="modal coder-wm-modal" style="width:min(640px,95vw);max-height:80vh;display:flex;flex-direction:column">
        <div class="modal-header">
          <span class="modal-title">Manage Workspaces</span>
          <button class="icon-btn small" id="coder-wm-close" title="Close">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>
        <div class="coder-wm-search-row" style="display:flex;gap:var(--space-xs);align-items:center">
          <input type="search" class="field-input coder-wm-search" id="coder-wm-search" placeholder="Filter by name or id…" autocomplete="off" style="flex:1">
          <div class="coder-wm-viewtoggle" style="display:flex;gap:2px">
            <button class="btn btn-sm" id="coder-wm-view-active" data-wm-view="active">Active</button>
            <button class="btn btn-sm btn-ghost" id="coder-wm-view-archived" data-wm-view="archived" title="Archived workspaces — restore or reclaim their disk space">Archived</button>
          </div>
        </div>
        <div class="modal-body coder-wm-body">
          <div class="coder-wm-list" id="coder-wm-list">
            <div class="coder-wm-empty"><p>Loading…</p></div>
          </div>
        </div>
        <div class="modal-footer coder-wm-footer">
          <button class="btn btn-ghost btn-sm" id="coder-wm-refresh" title="Reload the list">↻ Refresh</button>
          <div class="coder-wm-footer-right">
            <button class="btn btn-ghost btn-sm" id="coder-wm-import" title="Import a workspace from a .tar.gz export">⬆ Import</button>
            <button class="btn btn-primary btn-sm" id="coder-wm-new">+ New Workspace</button>
          </div>
          <input type="file" id="coder-wm-import-file" accept=".tar.gz,.tgz,application/gzip" hidden>
        </div>
      </div>`;
    document.body.appendChild(modal);

    modal.querySelector('#coder-wm-close').addEventListener('click', () => modal.classList.add('hidden'));
    modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.add('hidden'); });

    const listEl = modal.querySelector('#coder-wm-list');

    // Single delegated click handler — covers every per-row action via
    // data-wm-action attrs. Avoids re-wiring listeners after every
    // re-render.
    listEl.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-wm-action]');
      if (!btn) return;
      const action = btn.dataset.wmAction;
      const wsId = btn.dataset.wmId;
      const extra = {};
      if (action === 'toggle-always-on') extra.current = btn.dataset.wmCurrent;
      if (action === 'delete') extra.name = btn.dataset.wmName;
      if (action === 'purge') extra.name = btn.dataset.wmName;
      if (action === 'rename') extra.name = btn.dataset.wmName;
      _handleWorkspaceManagerAction(action, wsId, btn, listEl, extra);
    });

    const searchEl = modal.querySelector('#coder-wm-search');
    let searchDebounce = null;
    searchEl.addEventListener('input', () => {
      // 120ms debounce — re-render is cheap (server cache) but typing
      // fast still triggers visible flicker without a guard.
      clearTimeout(searchDebounce);
      searchDebounce = setTimeout(() => {
        _renderWorkspaceManagerList(listEl, searchEl.value);
      }, 120);
    });

    modal.querySelector('#coder-wm-refresh').addEventListener('click', () => {
      _renderWorkspaceManagerList(listEl, searchEl.value);
    });

    // Active / Archived segmented toggle.
    const activeBtn = modal.querySelector('#coder-wm-view-active');
    const archivedBtn = modal.querySelector('#coder-wm-view-archived');
    const applyView = (archived) => {
      _wmArchivedView = archived;
      activeBtn.className = archived ? 'btn btn-sm btn-ghost' : 'btn btn-sm';
      archivedBtn.className = archived ? 'btn btn-sm' : 'btn btn-sm btn-ghost';
      // Filter box only applies to the active list; hide it for archives.
      searchEl.style.visibility = archived ? 'hidden' : 'visible';
      _renderWorkspaceManagerList(listEl, searchEl.value);
    };
    activeBtn.addEventListener('click', () => applyView(false));
    archivedBtn.addEventListener('click', () => applyView(true));
    modal.querySelector('#coder-wm-new').addEventListener('click', () => {
      modal.classList.add('hidden');
      _openNewWorkspaceModal();
    });

    // Import: hidden file input triggered by the visible button. The
    // input is reset on every open so re-selecting the same file fires
    // the change event again (otherwise repeat-imports are silent).
    const importBtn = modal.querySelector('#coder-wm-import');
    const importInput = modal.querySelector('#coder-wm-import-file');
    importBtn.addEventListener('click', () => {
      importInput.value = '';
      importInput.click();
    });
    importInput.addEventListener('change', async () => {
      const file = importInput.files?.[0];
      if (!file) return;
      // Cheap client-side guard so the user gets feedback before a
      // 500MB upload starts.
      const MAX_BYTES = 500 * 1024 * 1024;
      if (file.size > MAX_BYTES) {
        showToast(`Archive exceeds ${MAX_BYTES / (1024 * 1024)}MB limit`, 'error', 6000);
        return;
      }
      // Strip the .tar.gz / .tgz extension for a sensible workspace name.
      const baseName = file.name.replace(/\.(tar\.gz|tgz)$/i, '') || 'imported';
      const originalLabel = importBtn.textContent;
      importBtn.textContent = 'Importing…';
      importBtn.disabled = true;
      try {
        const form = new FormData();
        form.append('name', baseName);
        form.append('tooling_profile', 'browser');
        form.append('archive', file);
        const resp = await fetch('/api/coder/workspaces/import', {
          method: 'POST',
          body: form,
        });
        if (!resp.ok) {
          let detail = '';
          try { detail = (await resp.json())?.error || ''; } catch (_) {}
          showToast(detail || `Import failed (HTTP ${resp.status})`, 'error', 6000);
          return;
        }
        const info = await resp.json();
        showToast(`Imported "${info.name}"`, 'success');
        await _renderWorkspaceManagerList(listEl, searchEl.value);
        // Refresh the workspace bar dropdown too — manager modal isn't
        // the only surface listing workspaces.
        _populateWorkspaceSelect();
      } catch (err) {
        showToast(`Import failed: ${err.message || 'network error'}`, 'error');
      } finally {
        importBtn.textContent = originalLabel;
        importBtn.disabled = false;
      }
    });
  }
  modal.classList.remove('hidden');
  // Always open on the Active view so the toggle state is predictable.
  _wmArchivedView = false;
  const activeBtn = modal.querySelector('#coder-wm-view-active');
  const archivedBtn = modal.querySelector('#coder-wm-view-archived');
  if (activeBtn) activeBtn.className = 'btn btn-sm';
  if (archivedBtn) archivedBtn.className = 'btn btn-sm btn-ghost';
  const listEl = modal.querySelector('#coder-wm-list');
  const searchEl = modal.querySelector('#coder-wm-search');
  if (searchEl) searchEl.style.visibility = 'visible';
  await _renderWorkspaceManagerList(listEl, searchEl?.value || '');
}

function _openGitSettingsModal() {
  let modal = document.getElementById('coder-git-settings-modal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'coder-git-settings-modal';
    modal.className = 'modal-overlay hidden';
    modal.innerHTML = `
      <div class="modal coder-git-settings-modal" style="width:min(520px,95vw)">
        <div class="modal-header">
          <span class="modal-title">Git Settings</span>
          <button class="icon-btn small" id="coder-git-settings-close">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>
        <div class="modal-body" style="display:flex;flex-direction:column;gap:var(--space-md)">
          <div class="field-group">
            <label class="field-label">Remote URL</label>
            <div style="display:flex;gap:var(--space-xs)">
              <input type="text" class="field-input" id="coder-git-remote-input" placeholder="https://github.com/user/repo.git or git@github.com:user/repo.git" style="flex:1">
              <button class="btn btn-sm btn-primary" id="coder-git-remote-save">Set</button>
            </div>
            <p class="field-hint" id="coder-git-remote-hint">Accepts HTTPS or SSH URLs. Auth uses the tokens below for HTTPS hosts.</p>
          </div>
          <div class="field-group">
            <label class="field-label">Git Tokens</label>
            <div id="coder-git-token-list"></div>
            <div style="display:flex;gap:var(--space-xs);margin-top:var(--space-xs);flex-wrap:wrap">
              <input type="text" class="field-input" id="coder-git-token-host" placeholder="github.com" style="width:140px">
              <input type="password" class="field-input" id="coder-git-token-value" placeholder="Token / PAT" style="flex:1;min-width:160px">
              <button class="btn btn-sm btn-primary" id="coder-git-token-add">Add</button>
            </div>
            <p class="field-hint">Stored encrypted (Fernet). The credential helper inside the workspace container reads them only over the Docker-internal bridge.</p>
          </div>
          <div class="field-group">
            <label class="field-label">Recent commits</label>
            <div id="coder-git-log-list" class="coder-git-log-list">
              <div class="coder-git-log-empty">No commits yet.</div>
            </div>
          </div>
        </div>
      </div>`;
    document.body.appendChild(modal);

    modal.querySelector('#coder-git-settings-close').addEventListener('click', () => modal.classList.add('hidden'));
    modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.add('hidden'); });

    modal.querySelector('#coder-git-remote-save').addEventListener('click', async () => {
      const input = modal.querySelector('#coder-git-remote-input');
      const hint = modal.querySelector('#coder-git-remote-hint');
      const url = input.value.trim();
      if (!currentCoder().workspaceId) return;
      const err = _validateRemoteUrl(url);
      if (err) {
        if (hint) {
          hint.textContent = err;
          hint.classList.add('field-hint-error');
        }
        input.focus();
        return;
      }
      if (hint) {
        hint.textContent = 'Accepts HTTPS or SSH URLs. Auth uses the tokens below for HTTPS hosts.';
        hint.classList.remove('field-hint-error');
      }
      try {
        const resp = await fetch(`/api/coder/workspaces/${currentCoder().workspaceId}/git/remote`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url }),
        });
        if (resp.ok) { showToast('Remote set', 'success'); _refreshGitHeaderStatus(); }
        else showToast('Failed to set remote', 'error');
      } catch (err) {
        showToast('Failed to set remote: ' + (err?.message || err), 'error');
        console.warn('[coder] set remote failed', err);
      }
    });

    modal.querySelector('#coder-git-token-add').addEventListener('click', async () => {
      const host = modal.querySelector('#coder-git-token-host').value.trim();
      const token = modal.querySelector('#coder-git-token-value').value.trim();
      if (!host || !token) { showToast('Enter host and token', 'warning'); return; }
      try {
        const resp = await fetch('/api/coder/git-tokens', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ host, token }),
        });
        if (resp.ok) {
          showToast(`Token saved for ${host}`, 'success');
          modal.querySelector('#coder-git-token-value').value = '';
          _loadGitTokenList();
        } else {
          showToast('Failed to save token', 'error');
        }
      } catch (err) {
        showToast('Failed to save token: ' + (err?.message || err), 'error');
        console.warn('[coder] save token failed', err);
      }
    });
  }

  // Populate current remote + recent log. Both come from the same
  // status endpoint — single fetch, two consumers.
  if (currentCoder().workspaceId) {
    fetch(`/api/coder/workspaces/${currentCoder().workspaceId}/git/status`)
      .then(r => r.json())
      .then(d => {
        const input = modal.querySelector('#coder-git-remote-input');
        if (d.remote && input) input.value = d.remote;
        _renderGitLog(modal, d.log || []);
      }).catch(() => {});
  }
  _loadGitTokenList();
  modal.classList.remove('hidden');
}

// Pre-flight check for the remote-URL input. Returns the empty string
// when the URL looks plausible, or a human-readable rejection. We don't
// try to authenticate or even contact the host — we just rule out the
// shapes that always fail (empty, whitespace, file paths, missing host).
function _validateRemoteUrl(url) {
  if (!url) return 'Enter a remote URL.';
  if (/\s/.test(url)) return 'URL must not contain whitespace.';
  // git@host:path
  if (url.startsWith('git@')) {
    const tail = url.slice(4);
    if (!tail.includes(':')) return 'SSH URL needs a ":" — e.g. git@github.com:user/repo.git';
    return '';
  }
  // ssh://user@host/...
  if (url.startsWith('ssh://')) {
    return '';
  }
  // https:// / http://
  if (url.startsWith('https://') || url.startsWith('http://')) {
    // Reject the bare scheme + ensure there's a host.
    const m = url.match(/^https?:\/\/([^/]+)/);
    if (!m || !m[1]) return 'URL needs a host — e.g. https://github.com/user/repo.git';
    return '';
  }
  // git:// (deprecated but still valid)
  if (url.startsWith('git://')) return '';
  // file:// is intentionally rejected — it works locally but is almost
  // always a misconfiguration (e.g. the user pasted a file system path).
  if (url.startsWith('file://')) {
    return 'file:// URLs are not supported. Use HTTPS or SSH to a real remote.';
  }
  return 'URL must start with https://, http://, ssh://, git://, or git@';
}

function _renderGitLog(modal, entries) {
  const list = modal.querySelector('#coder-git-log-list');
  if (!list) return;
  if (!entries.length) {
    list.innerHTML = '<div class="coder-git-log-empty">No commits yet — push your first commit to see it here.</div>';
    return;
  }
  // Entries are raw ``git log --oneline -5`` lines: "<sha7> <subject>".
  // Split once on the first space so subjects containing spaces stay
  // intact. Trim defensively in case the backend ever appends a tag
  // suffix or color codes.
  list.innerHTML = entries.map((line) => {
    const trimmed = (line || '').trim();
    const space = trimmed.indexOf(' ');
    const sha = space > 0 ? trimmed.slice(0, space) : trimmed;
    const subject = space > 0 ? trimmed.slice(space + 1) : '';
    return `
      <div class="coder-git-log-row">
        <span class="coder-git-log-sha">${escapeHtml(sha)}</span>
        <span class="coder-git-log-subject" title="${escapeHtml(subject)}">${escapeHtml(subject)}</span>
      </div>
    `;
  }).join('');
}

// ---------------------------------------------------------------------------
// Commit panel — stage / review / commit / (optionally) push.
//
// Replaces the old "Push = git add -A + auto-commit + push" footgun
// with a proper review flow. The user sees every changed file, ticks
// the ones to include, reads the diff, writes a real message, and
// chooses whether to push.
//
// State lives in module scope so a re-open after edits picks up where
// you left off (message + selected paths persist within a session).
// ---------------------------------------------------------------------------

const _commitPanel = {
  modal: null,
  selected: new Set(),
  draftMessage: '',
  files: [],          // [{path, status, staged}]
  activeDiffPath: '',
  loading: false,
};

async function _openCommitPanel() {
  let modal = document.getElementById('coder-commit-panel');
  if (!modal) {
    modal = _buildCommitPanelDOM();
    document.body.appendChild(modal);
  }
  _commitPanel.modal = modal;
  modal.classList.remove('hidden');
  await _refreshCommitPanel();
}

function _closeCommitPanel() {
  const modal = _commitPanel.modal;
  if (modal) modal.classList.add('hidden');
}

function _buildCommitPanelDOM() {
  const modal = document.createElement('div');
  modal.id = 'coder-commit-panel';
  modal.className = 'modal-overlay coder-commit-overlay hidden';
  modal.innerHTML = `
    <div class="coder-commit-modal" role="dialog" aria-modal="true" aria-label="Commit changes">
      <div class="coder-commit-header">
        <span class="coder-commit-title">Commit changes</span>
        <span class="coder-commit-branch" id="coder-commit-branch-label" title="Current branch"></span>
        <button class="icon-btn small" id="coder-commit-close-btn" title="Close" aria-label="Close">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
      <div class="coder-commit-body">
        <div class="coder-commit-files">
          <div class="coder-commit-files-toolbar">
            <button type="button" class="btn btn-sm" id="coder-commit-stage-all">Stage all</button>
            <button type="button" class="btn btn-sm" id="coder-commit-unstage-all">Unstage all</button>
            <button type="button" class="btn btn-sm" id="coder-commit-refresh" title="Re-read working tree">↻</button>
          </div>
          <div class="coder-commit-files-list" id="coder-commit-files-list">
            <div class="coder-commit-empty">Loading…</div>
          </div>
        </div>
        <div class="coder-commit-diff" id="coder-commit-diff-pane">
          <div class="coder-commit-diff-header" id="coder-commit-diff-header">No file selected.</div>
          <pre class="coder-commit-diff-body" id="coder-commit-diff-body"></pre>
        </div>
      </div>
      <div class="coder-commit-footer">
        <textarea
          class="coder-commit-message"
          id="coder-commit-message"
          rows="3"
          placeholder="Commit message — first line is the summary, blank line, then body."
          maxlength="2000"
        ></textarea>
        <div class="coder-commit-actions">
          <span class="coder-commit-summary" id="coder-commit-staged-summary">0 staged</span>
          <button type="button" class="btn btn-sm" id="coder-commit-do" disabled>Commit</button>
          <button type="button" class="btn btn-sm btn-primary" id="coder-commit-do-push" disabled>Commit and push</button>
        </div>
      </div>
    </div>
  `;
  modal.addEventListener('click', (ev) => { if (ev.target === modal) _closeCommitPanel(); });
  modal.querySelector('#coder-commit-close-btn').addEventListener('click', _closeCommitPanel);
  modal.querySelector('#coder-commit-refresh').addEventListener('click', _refreshCommitPanel);
  modal.querySelector('#coder-commit-stage-all').addEventListener('click', () => _bulkStage(true));
  modal.querySelector('#coder-commit-unstage-all').addEventListener('click', () => _bulkStage(false));
  modal.querySelector('#coder-commit-do').addEventListener('click', () => _doCommit({ push: false }));
  modal.querySelector('#coder-commit-do-push').addEventListener('click', () => _doCommit({ push: true }));

  const msg = modal.querySelector('#coder-commit-message');
  msg.addEventListener('input', () => {
    _commitPanel.draftMessage = msg.value;
    _updateCommitButtons();
  });

  // File row clicks (delegated): checkbox toggles stage state, name
  // shows diff. Single delegated handler to keep DOM cheap as files
  // re-render on every refresh.
  modal.querySelector('#coder-commit-files-list').addEventListener('click', (ev) => {
    const row = ev.target.closest('.coder-commit-file');
    if (!row) return;
    const path = row.dataset.path;
    if (!path) return;
    if (ev.target.closest('.coder-commit-file-toggle')) {
      _toggleFileStaged(path);
      return;
    }
    _showCommitDiff(path);
  });

  // Esc closes the panel — matches command palette / strategy popover.
  modal.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') _closeCommitPanel();
  });

  return modal;
}

async function _refreshCommitPanel() {
  if (!_commitPanel.modal || !currentCoder().workspaceId) return;
  _commitPanel.loading = true;
  const listEl = _commitPanel.modal.querySelector('#coder-commit-files-list');
  if (listEl) listEl.innerHTML = '<div class="coder-commit-empty">Loading…</div>';

  try {
    const [statusResp, headResp] = await Promise.all([
      fetch(`/api/coder/workspaces/${currentCoder().workspaceId}/git/file-status`, { credentials: 'include' }),
      fetch(`/api/coder/workspaces/${currentCoder().workspaceId}/git/status`, { credentials: 'include' }),
    ]);
    const statusData = statusResp.ok ? await statusResp.json() : { files: [] };
    const headData = headResp.ok ? await headResp.json() : {};

    // Mark each file as staged if its porcelain X (index column) is set.
    // We need the raw porcelain to know index vs worktree, so re-derive
    // from the dedicated /staged-paths probe rather than the collapsed
    // status the decorations endpoint returns. Two calls, one DOM
    // refresh.
    const stagedResp = await fetch(
      `/api/coder/workspaces/${currentCoder().workspaceId}/git/diff?staged=1`,
      { credentials: 'include' },
    );
    const stagedData = stagedResp.ok ? await stagedResp.json() : { diff: '' };
    const stagedPaths = _parseDiffFilenames(stagedData.diff || '');

    _commitPanel.files = (statusData.files || []).map((f) => ({
      path: f.path,
      status: f.status,
      staged: stagedPaths.has(f.path),
    }));

    // Pre-select files the agent recently modified (priority categories
    // M/A/D) IF the user hasn't already touched the selection in this
    // session. Otherwise honour their current selection.
    if (_commitPanel.selected.size === 0) {
      for (const f of _commitPanel.files) {
        if (f.staged) _commitPanel.selected.add(f.path);
      }
    }

    const branchLabel = _commitPanel.modal.querySelector('#coder-commit-branch-label');
    if (branchLabel) branchLabel.textContent = headData.branch ? `on ${headData.branch}` : '';

    _renderCommitFileList();
    _updateCommitButtons();

    // Auto-show diff for the first file if nothing's selected yet.
    if (!_commitPanel.activeDiffPath && _commitPanel.files.length) {
      _showCommitDiff(_commitPanel.files[0].path);
    } else if (_commitPanel.activeDiffPath) {
      // Refresh the open diff in case its contents shifted.
      _showCommitDiff(_commitPanel.activeDiffPath);
    }
  } catch (err) {
    console.warn('Commit panel refresh failed', err);
    if (listEl) listEl.innerHTML = '<div class="coder-commit-empty">Failed to load changes.</div>';
  } finally {
    _commitPanel.loading = false;
  }
}

function _parseDiffFilenames(diffText) {
  // ``diff --git a/foo b/bar`` lines mark each file. We collect ``b/``
  // paths (post-rename / new path) so the staged set matches what
  // ends up in HEAD.
  const set = new Set();
  for (const line of (diffText || '').split('\n')) {
    if (!line.startsWith('diff --git')) continue;
    const m = line.match(/diff --git a\/(.+?) b\/(.+)$/);
    if (!m) continue;
    set.add('/workspace/' + m[2].trim());
  }
  return set;
}

function _renderCommitFileList() {
  const listEl = _commitPanel.modal?.querySelector('#coder-commit-files-list');
  if (!listEl) return;
  const files = _commitPanel.files;
  if (!files.length) {
    listEl.innerHTML = '<div class="coder-commit-empty">Nothing to commit — your working tree is clean.</div>';
    return;
  }
  listEl.innerHTML = files.map((f) => {
    const checked = _commitPanel.selected.has(f.path) ? 'checked' : '';
    const active = f.path === _commitPanel.activeDiffPath ? 'active' : '';
    const label = (f.path || '').replace(/^\/workspace\//, '');
    return `
      <div class="coder-commit-file ${active}" data-path="${escapeHtml(f.path)}">
        <label class="coder-commit-file-toggle" title="Stage / unstage">
          <input type="checkbox" ${checked} data-path="${escapeHtml(f.path)}">
        </label>
        <span class="coder-commit-file-status" data-status="${escapeHtml(f.status)}" title="${escapeHtml(_gitStatusLabel(f.status))}">${escapeHtml(f.status)}</span>
        <span class="coder-commit-file-name" title="${escapeHtml(label)}">${escapeHtml(label)}</span>
      </div>
    `;
  }).join('');
}

async function _toggleFileStaged(path) {
  if (!path) return;
  const wasSelected = _commitPanel.selected.has(path);
  // Optimistically flip the local state so the UI feels responsive,
  // then call the backend; revert on failure.
  if (wasSelected) _commitPanel.selected.delete(path);
  else _commitPanel.selected.add(path);
  _renderCommitFileList();
  _updateCommitButtons();

  const endpoint = wasSelected ? 'unstage' : 'stage';
  try {
    const resp = await fetch(
      `/api/coder/workspaces/${currentCoder().workspaceId}/git/${endpoint}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paths: [path] }),
      },
    );
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      showToast(extractErrorMessage(err, `${endpoint} failed`), 'error');
      // Revert.
      if (wasSelected) _commitPanel.selected.add(path);
      else _commitPanel.selected.delete(path);
      _renderCommitFileList();
      _updateCommitButtons();
    }
  } catch {
    showToast(`${endpoint} failed`, 'error');
    if (wasSelected) _commitPanel.selected.add(path);
    else _commitPanel.selected.delete(path);
    _renderCommitFileList();
    _updateCommitButtons();
  }
}

async function _bulkStage(stage) {
  const paths = _commitPanel.files.map((f) => f.path);
  if (!paths.length) return;
  try {
    const resp = await fetch(
      `/api/coder/workspaces/${currentCoder().workspaceId}/git/${stage ? 'stage' : 'unstage'}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paths }),
      },
    );
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      showToast(extractErrorMessage(err, 'Bulk action failed'), 'error');
      return;
    }
    _commitPanel.selected = stage ? new Set(paths) : new Set();
    _renderCommitFileList();
    _updateCommitButtons();
    showToast(stage ? 'All files staged' : 'All files unstaged', 'success');
  } catch {
    showToast('Bulk action failed', 'error');
  }
}

/**
 * Fetch a file's staged + unstaged diff segments. Shared spine for the
 * commit panel's diff pane AND the standalone "View changes" modal so
 * the two never drift. Returns `{ segments: [{label, body}] }` — empty
 * segments means the file is untracked (no git diff exists yet).
 */
async function _fetchFileDiffSegments(workspaceId, path) {
  const [stagedResp, unstagedResp] = await Promise.all([
    fetch(
      `/api/coder/workspaces/${encodeURIComponent(workspaceId)}/git/diff?staged=1&path=${encodeURIComponent(path)}`,
      { credentials: 'include' },
    ),
    fetch(
      `/api/coder/workspaces/${encodeURIComponent(workspaceId)}/git/diff?path=${encodeURIComponent(path)}`,
      { credentials: 'include' },
    ),
  ]);
  const stagedData = stagedResp.ok ? await stagedResp.json() : { diff: '' };
  const unstagedData = unstagedResp.ok ? await unstagedResp.json() : { diff: '' };
  const stagedDiff = (stagedData.diff || '').trim();
  const unstagedDiff = (unstagedData.diff || '').trim();
  const segments = [];
  if (stagedDiff) segments.push({ label: 'Staged (about to commit)', body: stagedDiff });
  if (unstagedDiff) segments.push({ label: 'Unstaged (not in this commit)', body: unstagedDiff });
  return { segments };
}

async function _showCommitDiff(path) {
  _commitPanel.activeDiffPath = path;
  _renderCommitFileList();
  const header = _commitPanel.modal?.querySelector('#coder-commit-diff-header');
  const body = _commitPanel.modal?.querySelector('#coder-commit-diff-body');
  if (!header || !body) return;
  const rel = (path || '').replace(/^\/workspace\//, '');
  header.textContent = rel;
  body.textContent = 'Loading diff…';
  try {
    // Staged wins when both exist because the user is reviewing what
    // they're about to commit. Shared fetch spine with the standalone
    // diff modal (_fetchFileDiffSegments).
    const { segments } = await _fetchFileDiffSegments(currentCoder().workspaceId, path);
    if (!segments.length) {
      // Untracked files don't show in either diff; surface the file
      // as a new addition so the panel doesn't read as broken.
      body.innerHTML = `<span class="coder-commit-diff-untracked">Untracked — no diff yet. Stage to include the entire file as a new addition.</span>`;
      return;
    }
    body.innerHTML = segments.map((s) => `
      <div class="coder-commit-diff-segment-label">${escapeHtml(s.label)}</div>
      <div class="coder-commit-diff-segment">${_renderDiffHtml(s.body)}</div>
    `).join('');
  } catch {
    body.textContent = 'Failed to load diff.';
  }
}

// ---------------------------------------------------------------------------
// Standalone per-file diff modal — "View changes" from the file tree.
// Reuses the commit panel's diff renderer + fetch spine so a user can
// review what changed in ONE file at any time, without opening the
// full commit staging flow. For untracked (new) files — the common
// case when a small model sprays a fresh script — it falls back to
// rendering the whole file as an addition so it's still reviewable.
// ---------------------------------------------------------------------------

function _buildFileDiffModal() {
  const modal = document.createElement('div');
  modal.id = 'coder-file-diff-modal';
  modal.className = 'modal-overlay coder-diff-overlay hidden';
  modal.innerHTML = `
    <div class="coder-diff-modal" role="dialog" aria-modal="true" aria-label="File changes">
      <div class="coder-diff-header">
        <span class="coder-diff-title" id="coder-file-diff-title"></span>
        <div class="coder-diff-header-actions">
          <button class="btn btn-sm" id="coder-file-diff-open" title="Open this file in the editor">Open file</button>
          <button class="icon-btn small" id="coder-file-diff-close" title="Close (Esc)" aria-label="Close">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>
      </div>
      <pre class="coder-diff-body coder-commit-diff-body" id="coder-file-diff-body"></pre>
    </div>
  `;
  const close = () => modal.classList.add('hidden');
  modal.addEventListener('click', (ev) => { if (ev.target === modal) close(); });
  modal.querySelector('#coder-file-diff-close').addEventListener('click', close);
  modal.addEventListener('keydown', (ev) => { if (ev.key === 'Escape') close(); });
  modal.querySelector('#coder-file-diff-open').addEventListener('click', () => {
    const path = modal.dataset.path;
    const name = modal.dataset.name;
    if (path) _openFileInEditor(currentCoder().workspaceId, path, name);
    close();
  });
  return modal;
}

// Commit-history diff modal — shows what changed IN a checkpoint/commit
// (git show), turning the checkpoints list into a browsable history. Reuses
// the same overlay/renderer as the per-file diff. View-only; the list's
// Revert button remains the restore path.
function _buildCommitDiffModalEl() {
  const modal = document.createElement('div');
  modal.id = 'coder-commit-diff-modal';
  modal.className = 'modal-overlay coder-diff-overlay hidden';
  modal.innerHTML = `
    <div class="coder-diff-modal" role="dialog" aria-modal="true" aria-label="Commit changes">
      <div class="coder-diff-header">
        <div class="coder-commit-diff-heading">
          <span class="coder-diff-title" id="coder-commit-diff-title"></span>
          <span class="coder-commit-diff-sub" id="coder-commit-diff-sub"></span>
        </div>
        <button class="icon-btn small" id="coder-commit-diff-close" title="Close (Esc)" aria-label="Close">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
      <pre class="coder-diff-body coder-commit-diff-body" id="coder-commit-diff-modal-body"></pre>
    </div>
  `;
  const close = () => modal.classList.add('hidden');
  modal.addEventListener('click', (ev) => { if (ev.target === modal) close(); });
  modal.querySelector('#coder-commit-diff-close').addEventListener('click', close);
  modal.addEventListener('keydown', (ev) => { if (ev.key === 'Escape') close(); });
  return modal;
}

async function _showCommitDiffModal(hash, subject) {
  if (!hash || !currentCoder().workspaceId) return;
  let modal = document.getElementById('coder-commit-diff-modal');
  if (!modal) { modal = _buildCommitDiffModalEl(); document.body.appendChild(modal); }
  modal.dataset.hash = hash;
  modal.classList.remove('hidden');
  const title = modal.querySelector('#coder-commit-diff-title');
  const sub = modal.querySelector('#coder-commit-diff-sub');
  const body = modal.querySelector('#coder-commit-diff-modal-body');
  title.textContent = subject || hash;
  sub.textContent = hash;
  body.textContent = 'Loading changes…';
  try {
    const resp = await fetch(
      `/api/coder/checkpoints/${encodeURIComponent(currentCoder().workspaceId)}/show?hash=${encodeURIComponent(hash)}`,
    );
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) { body.textContent = data.error || 'Failed to load commit.'; return; }
    const meta = data.meta || {};
    const when = meta.timestamp ? new Date(meta.timestamp * 1000).toLocaleString() : '';
    sub.textContent = [hash, meta.author, when].filter(Boolean).join(' · ');
    const diff = (data.diff || '').trim();
    body.innerHTML = diff
      ? _renderDiffHtml(diff)
      : '<span class="coder-commit-diff-untracked">No file changes in this commit.</span>';
  } catch {
    body.textContent = 'Failed to load commit.';
  }
}

async function _showFileDiff(workspaceId, filePath, fileName) {
  let modal = document.getElementById('coder-file-diff-modal');
  if (!modal) { modal = _buildFileDiffModal(); document.body.appendChild(modal); }
  modal.dataset.path = filePath;
  modal.dataset.name = fileName;
  modal.classList.remove('hidden');
  const title = modal.querySelector('#coder-file-diff-title');
  const body = modal.querySelector('#coder-file-diff-body');
  title.textContent = (filePath || '').replace(/^\/workspace\//, '');
  body.textContent = 'Loading diff…';
  try {
    const { segments } = await _fetchFileDiffSegments(workspaceId, filePath);
    if (segments.length) {
      body.innerHTML = segments.map((s) => `
        <div class="coder-commit-diff-segment-label">${escapeHtml(s.label)}</div>
        <div class="coder-commit-diff-segment">${_renderDiffHtml(s.body)}</div>
      `).join('');
      return;
    }
    // No git diff — either an untracked new file or a clean file. Read
    // the content and, if non-empty, render it as an all-additions view
    // so a freshly-sprayed script is reviewable here too.
    const readResp = await fetch(
      `/api/coder/files/${encodeURIComponent(workspaceId)}/read?path=${encodeURIComponent(filePath)}`,
    );
    const content = readResp.ok ? ((await readResp.json()).content || '') : '';
    if (!content.trim()) {
      body.innerHTML = `<span class="coder-commit-diff-untracked">No changes to show — this file matches the last commit.</span>`;
      return;
    }
    const addBody = content.split('\n').map((l) => '+' + l).join('\n');
    body.innerHTML =
      `<div class="coder-commit-diff-segment-label">New file (untracked — shown as a full addition)</div>` +
      `<div class="coder-commit-diff-segment">${_renderDiffHtml(addBody)}</div>`;
  } catch {
    body.textContent = 'Failed to load diff.';
  }
}

function _renderDiffHtml(text) {
  // Lightweight syntax: + lines green, - lines red, @@ headers muted.
  // We don't try to do full intraline highlighting — the colored
  // gutter line is enough context for review.
  return text.split('\n').map((line) => {
    const safe = escapeHtml(line);
    if (line.startsWith('+++') || line.startsWith('---') || line.startsWith('diff ')) {
      return `<span class="diff-hdr">${safe}</span>`;
    }
    if (line.startsWith('@@')) return `<span class="diff-hunk">${safe}</span>`;
    if (line.startsWith('+')) return `<span class="diff-add">${safe}</span>`;
    if (line.startsWith('-')) return `<span class="diff-del">${safe}</span>`;
    return safe;
  }).join('\n');
}

function _updateCommitButtons() {
  const modal = _commitPanel.modal;
  if (!modal) return;
  const summary = modal.querySelector('#coder-commit-staged-summary');
  const doBtn = modal.querySelector('#coder-commit-do');
  const doPushBtn = modal.querySelector('#coder-commit-do-push');
  const msg = modal.querySelector('#coder-commit-message');
  const staged = _commitPanel.selected.size;
  const hasMessage = msg && msg.value.trim().length > 0;
  const ready = staged > 0 && hasMessage;
  if (summary) {
    summary.textContent = staged === 0
      ? '0 staged'
      : `${staged} file${staged === 1 ? '' : 's'} staged`;
  }
  if (doBtn) doBtn.disabled = !ready;
  if (doPushBtn) doPushBtn.disabled = !ready;
}

async function _doCommit({ push }) {
  const msg = (_commitPanel.modal?.querySelector('#coder-commit-message').value || '').trim();
  if (!msg) { showToast('Commit message is required', 'warning'); return; }
  if (!_commitPanel.selected.size) { showToast('No files staged', 'warning'); return; }

  const doBtn = _commitPanel.modal?.querySelector('#coder-commit-do');
  const doPushBtn = _commitPanel.modal?.querySelector('#coder-commit-do-push');
  if (doBtn) doBtn.disabled = true;
  if (doPushBtn) doPushBtn.disabled = true;
  showToast(push ? 'Committing and pushing…' : 'Committing…', 'info', 3000);

  try {
    const commitResp = await fetch(
      `/api/coder/workspaces/${currentCoder().workspaceId}/git/commit`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg }),
      },
    );
    if (!commitResp.ok) {
      const err = await commitResp.json().catch(() => ({}));
      showToast(extractErrorMessage(err, 'Commit failed'), 'error');
      _updateCommitButtons();
      return;
    }

    if (push) {
      const pushResp = await fetch(
        `/api/coder/workspaces/${currentCoder().workspaceId}/git/push`,
        { method: 'POST' },
      );
      const pushData = await pushResp.json().catch(() => ({}));
      if (!pushResp.ok) {
        // Commit succeeded; push didn't. Tell the user both.
        showToast(`Committed locally, but push failed: ${pushData.error || 'unknown'}`, 'error', 8000);
        if (pushResp.status === 401) _openGitSettingsModal();
        await _refreshGitHeaderStatus();
        _refreshCommitPanel();
        return;
      }
      showToast('Committed and pushed', 'success');
    } else {
      showToast('Committed', 'success');
    }

    // Reset local state for the next commit and refresh views.
    _commitPanel.selected = new Set();
    _commitPanel.draftMessage = '';
    if (_commitPanel.modal) {
      const mt = _commitPanel.modal.querySelector('#coder-commit-message');
      if (mt) mt.value = '';
    }
    await _refreshGitHeaderStatus();
    await _refreshCommitPanel();
    await _populateFileTree(currentCoder().workspaceId);
  } catch (err) {
    showToast('Commit failed: ' + (err?.message || err), 'error');
  } finally {
    _updateCommitButtons();
  }
}

// ---------------------------------------------------------------------------
// Branch picker — anchored popover off the branch chip. Lists local
// branches with last-commit subject + upstream, supports switching
// (with a server-side dirty-tree guard) and creating new branches.
// ---------------------------------------------------------------------------

let _branchPopover = null;
let _branchPopoverAnchor = null;
let _branchPopoverOutsideHandler = null;

async function _toggleBranchPopover(anchor) {
  if (_branchPopover) {
    _closeBranchPopover();
    return;
  }
  await _openBranchPopover(anchor);
}

async function _openBranchPopover(anchor) {
  if (!anchor || !currentCoder().workspaceId) return;
  const pop = document.createElement('div');
  pop.className = 'coder-branch-popover';
  pop.setAttribute('role', 'menu');
  pop.innerHTML = `
    <div class="coder-branch-popover-header">Branches</div>
    <div class="coder-branch-popover-list" id="coder-branch-popover-list">
      <div class="coder-branch-popover-empty">Loading…</div>
    </div>
    <div class="coder-branch-popover-divider"></div>
    <div class="coder-branch-popover-new">
      <input type="text" id="coder-branch-popover-new-input" placeholder="New branch name…" maxlength="128" />
      <button type="button" class="btn btn-sm btn-primary" id="coder-branch-popover-new-create">Create</button>
    </div>
    <div class="coder-branch-popover-footer">
      Switching warns when your working tree has uncommitted changes.
    </div>
  `;
  document.body.appendChild(pop);
  _branchPopover = pop;
  _branchPopoverAnchor = anchor;
  _positionBranchPopover();

  pop.querySelector('#coder-branch-popover-list').addEventListener('click', (ev) => {
    const btn = ev.target.closest('.coder-branch-row');
    if (!btn) return;
    const name = btn.dataset.branch;
    if (!name) return;
    _switchBranch(name);
  });

  const newInput = pop.querySelector('#coder-branch-popover-new-input');
  const newBtn = pop.querySelector('#coder-branch-popover-new-create');
  const submit = () => _createBranch(newInput.value);
  newBtn.addEventListener('click', submit);
  newInput.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') { ev.preventDefault(); submit(); }
    if (ev.key === 'Escape') { ev.preventDefault(); _closeBranchPopover(); }
  });

  _branchPopoverOutsideHandler = (ev) => {
    if (!_branchPopover) return;
    if (_branchPopover.contains(ev.target)) return;
    if (anchor.contains(ev.target)) return;
    _closeBranchPopover();
  };
  document.addEventListener('click', _branchPopoverOutsideHandler);
  document.addEventListener('keydown', _branchPopoverKeydown);

  await _loadBranchList();
}

function _branchPopoverKeydown(ev) {
  if (ev.key === 'Escape' && _branchPopover) {
    ev.preventDefault();
    _closeBranchPopover();
  }
}

function _closeBranchPopover() {
  if (_branchPopover?.parentNode) _branchPopover.parentNode.removeChild(_branchPopover);
  _branchPopover = null;
  _branchPopoverAnchor = null;
  if (_branchPopoverOutsideHandler) {
    document.removeEventListener('click', _branchPopoverOutsideHandler);
    _branchPopoverOutsideHandler = null;
  }
  document.removeEventListener('keydown', _branchPopoverKeydown);
}

function _positionBranchPopover() {
  if (!_branchPopover || !_branchPopoverAnchor) return;
  const rect = _branchPopoverAnchor.getBoundingClientRect();
  const pop = _branchPopover;
  pop.style.position = 'fixed';
  pop.style.zIndex = '9500';
  pop.style.visibility = 'hidden';
  pop.style.left = '0';
  pop.style.top = '0';
  const popRect = pop.getBoundingClientRect();
  const wantLeft = Math.max(8, rect.left);
  const wantTop = rect.bottom + 6;
  const overflowRight = wantLeft + popRect.width - window.innerWidth + 8;
  const adjLeft = overflowRight > 0 ? wantLeft - overflowRight : wantLeft;
  const overflowBottom = wantTop + popRect.height - window.innerHeight + 8;
  const adjTop = overflowBottom > 0 ? rect.top - popRect.height - 6 : wantTop;
  pop.style.left = `${adjLeft}px`;
  pop.style.top = `${adjTop}px`;
  pop.style.visibility = 'visible';
}

async function _loadBranchList() {
  const listEl = _branchPopover?.querySelector('#coder-branch-popover-list');
  if (!listEl) return;
  try {
    const resp = await fetch(
      `/api/coder/workspaces/${currentCoder().workspaceId}/git/branches`,
      { credentials: 'include' },
    );
    if (!resp.ok) {
      listEl.innerHTML = '<div class="coder-branch-popover-empty">Failed to load branches.</div>';
      return;
    }
    const data = await resp.json();
    const branches = Array.isArray(data.branches) ? data.branches : [];
    if (!branches.length) {
      listEl.innerHTML = '<div class="coder-branch-popover-empty">No branches yet.</div>';
      return;
    }
    listEl.innerHTML = branches.map((b) => {
      const upstream = b.upstream
        ? `<span class="coder-branch-row-upstream" title="Tracks ${escapeHtml(b.upstream)}">↪ ${escapeHtml(b.upstream)}</span>`
        : '';
      const subject = b.subject
        ? `<span class="coder-branch-row-subject" title="${escapeHtml(b.subject)}">${escapeHtml(b.subject)}</span>`
        : '';
      return `
        <button type="button" class="coder-branch-row ${b.current ? 'current' : ''}" data-branch="${escapeHtml(b.name)}" role="menuitem">
          <span class="coder-branch-row-check">${b.current ? '✓' : ''}</span>
          <span class="coder-branch-row-body">
            <span class="coder-branch-row-name">${escapeHtml(b.name)}</span>
            ${subject}
          </span>
          ${upstream}
        </button>
      `;
    }).join('');
    _positionBranchPopover();
  } catch {
    listEl.innerHTML = '<div class="coder-branch-popover-empty">Failed to load branches.</div>';
  }
}

async function _switchBranch(name) {
  if (!name || !currentCoder().workspaceId) return;
  if (_lastGitStatus?.branch === name) {
    _closeBranchPopover();
    return;
  }
  try {
    const resp = await fetch(
      `/api/coder/workspaces/${currentCoder().workspaceId}/git/checkout`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ branch: name }),
      },
    );
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      if (err.error_code === 'dirty_tree') {
        showToast(err.error, 'warning', 8000);
      } else {
        showToast(extractErrorMessage(err, 'Switch failed'), 'error');
      }
      return;
    }
    showToast(`Switched to ${name}`, 'success');
    _closeBranchPopover();
    await _refreshGitHeaderStatus();
    await _populateFileTree(currentCoder().workspaceId);
  } catch (err) {
    showToast('Switch failed: ' + (err?.message || err), 'error');
  }
}

async function _createBranch(rawName) {
  const name = (rawName || '').trim();
  if (!name) { showToast('Branch name required', 'warning'); return; }
  try {
    const resp = await fetch(
      `/api/coder/workspaces/${currentCoder().workspaceId}/git/branch`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      },
    );
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      showToast(extractErrorMessage(err, 'Create failed'), 'error');
      return;
    }
    showToast(`Branch ${name} created`, 'success');
    _closeBranchPopover();
    await _refreshGitHeaderStatus();
  } catch (err) {
    showToast('Create failed: ' + (err?.message || err), 'error');
  }
}

async function _loadGitTokenList() {
  const container = document.getElementById('coder-git-token-list');
  if (!container) return;
  try {
    const resp = await fetch('/api/coder/git-tokens');
    const data = await resp.json();
    if (!data.tokens?.length) {
      container.innerHTML = '<div style="font-size:var(--text-xs);color:var(--text-muted)">No tokens configured</div>';
      return;
    }
    container.innerHTML = data.tokens.map(t => `
      <div style="display:flex;align-items:center;gap:var(--space-xs);padding:4px 0">
        <span style="font-size:var(--text-xs);flex:1">${escapeHtml(t.host)}</span>
        <span style="font-size:var(--text-xs);color:var(--text-muted)">${escapeHtml(t.username || 'oauth2')}</span>
        <button class="btn btn-sm" data-delete-host="${escapeHtml(t.host)}" style="color:var(--error);font-size:var(--text-xs);padding:2px 6px">Remove</button>
      </div>
    `).join('');
    container.querySelectorAll('[data-delete-host]').forEach(btn => {
      btn.addEventListener('click', async () => {
        await fetch(`/api/coder/git-tokens/${encodeURIComponent(btn.dataset.deleteHost)}`, { method: 'DELETE' });
        showToast('Token removed', 'success');
        _loadGitTokenList();
      });
    });
  } catch { container.innerHTML = ''; }
}

function _scheduleFileTreeRefresh() {
  clearTimeout(currentCoder().fileTreeRefreshTimer);
  currentCoder().fileTreeRefreshTimer = setTimeout(() => {
    if (currentCoder().workspaceId) _populateFileTree(currentCoder().workspaceId);
    // File tree mutations (turn end, manual refresh) invalidate the
    // command palette's file index so the next palette open re-fetches
    // /files-flat. Lazy invalidation — we don't preemptively refetch
    // because the palette is rarely open.
    import('./command-palette.js').then((mod) => {
      mod.invalidateFilesCache?.();
    }).catch(() => { /* palette not loaded yet — no cache to clear */ });
  }, 500);
}

// Register the coder mode's actions with the command palette. Each
// entry is independently runnable, with a ``when`` guard so a
// workspace-scoped action doesn't appear when no workspace is active.
//
// Ordering inside the palette is by fuzzy-score, not by registration
// order — so adding a new action here doesn't disrupt existing
// muscle memory.
function _registerCoderCommands(register) {
  const hasWorkspace = () => Boolean(currentCoder().workspaceId);
  register({
    id: 'coder.file.new',
    label: 'New file…',
    hint: 'Create a file at the workspace root',
    group: 'Files',
    keywords: 'create add touch',
    when: hasWorkspace,
    run: () => _createNewFileIn(currentCoder().workspaceId, '/workspace'),
  });
  register({
    id: 'coder.file.newFolder',
    label: 'New folder…',
    hint: 'Create a folder at the workspace root',
    group: 'Files',
    keywords: 'directory mkdir',
    when: hasWorkspace,
    run: () => _createNewFolderIn(currentCoder().workspaceId, '/workspace'),
  });
  register({
    id: 'coder.file.refresh',
    label: 'Refresh file tree',
    hint: 'Re-read the workspace from disk',
    group: 'Files',
    keywords: 'reload sync',
    when: hasWorkspace,
    run: () => _populateFileTree(currentCoder().workspaceId),
  });
  register({
    id: 'coder.run.cancel',
    label: 'Cancel current run',
    hint: 'Stop the agent mid-turn',
    group: 'Run',
    keywords: 'stop halt abort esc',
    when: hasWorkspace,
    run: () => document.getElementById('coder-inspector-cancel-btn')?.click(),
  });
  register({
    id: 'coder.workspace.new',
    label: 'New workspace…',
    hint: 'Create a fresh container',
    group: 'Workspace',
    keywords: 'create add',
    run: () => document.getElementById('coder-add-workspace-btn')?.click(),
  });
  register({
    id: 'coder.workspace.export',
    label: 'Export workspace',
    hint: 'Download as .tar.gz',
    group: 'Workspace',
    keywords: 'download backup',
    when: hasWorkspace,
    run: () => document.getElementById('coder-export-workspace-btn')?.click(),
  });
  register({
    id: 'coder.workspace.safeguards',
    label: 'Toggle safeguards',
    hint: 'Enable/disable soft circuit-breakers',
    group: 'Workspace',
    keywords: 'limits breakers',
    when: hasWorkspace,
    run: () => document.getElementById('coder-safeguards-btn')?.click(),
  });
  register({
    id: 'coder.git.commit',
    label: 'Commit changes…',
    hint: 'Stage, review, and commit (optionally push)',
    group: 'Git',
    keywords: 'stage diff message',
    when: hasWorkspace,
    run: () => document.getElementById('coder-git-commit-btn')?.click(),
  });
  register({
    id: 'coder.git.switchBranch',
    label: 'Switch branch…',
    hint: 'Open the branch picker',
    group: 'Git',
    keywords: 'checkout',
    when: hasWorkspace,
    run: () => document.getElementById('coder-git-branch')?.click(),
  });
  register({
    id: 'coder.git.pull',
    label: 'Git pull',
    hint: 'Fetch + merge from origin',
    group: 'Git',
    keywords: 'sync update',
    when: hasWorkspace,
    run: () => document.getElementById('coder-git-pull-btn')?.click(),
  });
  register({
    id: 'coder.git.push',
    label: 'Git push',
    hint: 'Auto-commit and push',
    group: 'Git',
    keywords: 'sync upload',
    when: hasWorkspace,
    run: () => document.getElementById('coder-git-push-btn')?.click(),
  });
  register({
    id: 'coder.git.settings',
    label: 'Git settings',
    hint: 'Configure remote + credentials',
    group: 'Git',
    keywords: 'remote origin auth',
    when: hasWorkspace,
    run: () => document.getElementById('coder-git-settings-btn')?.click(),
  });
  register({
    id: 'coder.preview.open',
    label: 'Open preview',
    hint: 'Show the workspace preview pane',
    group: 'View',
    keywords: 'browser dev server',
    when: hasWorkspace,
    run: () => currentCoder().dom.previewToggleBtn?.click() || currentCoder().dom.workbenchPreviewTab?.click(),
  });
  register({
    id: 'coder.view.toggle',
    label: 'Toggle terminal / editor',
    hint: 'Mobile two-pane swap',
    group: 'View',
    keywords: 'swap layout',
    when: hasWorkspace,
    run: () => toggleView(),
  });
  register({
    id: 'coder.exit',
    label: 'Exit coder mode',
    hint: 'Back to chat',
    group: 'View',
    keywords: 'leave close',
    run: () => document.getElementById('coder-exit-btn')?.click(),
  });
}

/**
 * Toggle between terminal and editor view (mobile).
 */
export function toggleView() {
  const app = document.getElementById('app');
  const current = app?.getAttribute('data-coder-view') || 'terminal';
  const next = current === 'terminal' ? 'editor' : 'terminal';
  app?.setAttribute('data-coder-view', next);

  if (next === 'terminal' && currentCoder().terminalId) {
    requestAnimationFrame(() => Terminal.fit(currentCoder().terminalId));
  }
}
