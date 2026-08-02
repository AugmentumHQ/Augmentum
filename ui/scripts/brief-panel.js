/**
 * brief-panel.js — the companion "brief": a right-side panel that presents a
 * COMPLETED coder run so the user can act on it. Reuses companion-detail.js's
 * shell (open/replace/back/actions/note) via its ``mount()`` hook and injects
 * a LIVE hero (running dev-server preview) + the code diff (coder-review's
 * ``mountReviewPanel``) into the body.
 *
 * Consumes the ``coder_run_completed`` envelope the backend emits
 * (jobs/handlers/coder_background_run.py::_emit_run_perception):
 *   { ok, kind:'coder_run', workspace_id, run_id, review_turn_id,
 *     prompt, answer_preview, tool_calls, elapsed_s, failure? }
 *
 * Honesty guard (spec 2026-07-27): the verdict note reflects what was actually
 * verified. The oracle verdict lands in Phase 3; until then we state
 * "reported complete, not independently verified — review the diff" rather
 * than implying success. Citations (drill-into-code dropdowns) are Phase 3;
 * the diff below is the MVP evidence surface.
 *
 * Test without a backend completion (dev console):
 *   __previewBrief()               // stub envelope
 *   __previewBrief({ ...envelope }) // your own
 * Point the hero/diff at a real workspace first:
 *   window.__briefDemoWs = 'ws_...'; window.__briefDemoTurn = 'ctr_...';
 */

import { openCompanionDetail, closeCompanionDetail } from './companion-detail.js';
import { escapeHtml } from './app.js';

const STYLE_ID = 'brief-panel-style';

function _ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = `
    .brief-hero-frame {
      position: absolute; inset: 0; width: 100%; height: 100%;
      border: 0; background: #0d0f14; z-index: 1;
    }
    .brief-hero-idle {
      position: absolute; inset: 0; display: flex;
      align-items: center; justify-content: center; text-align: center;
      padding: 12px; font-size: 12px; z-index: 1;
      color: var(--text-secondary, #9aa0a6);
      background: color-mix(in srgb, currentColor 6%, transparent);
    }
    .brief-section { margin-top: 16px; }
    .brief-section-title {
      font-size: 11px; font-weight: 700; letter-spacing: 0.4px;
      text-transform: uppercase; color: var(--text-secondary, #9aa0a6);
      margin-bottom: 8px;
    }
    .brief-section .coder-review-card,
    .brief-section > * { max-width: 100%; }
    .brief-empty { font-size: 12.5px; color: var(--text-secondary, #9aa0a6); }
    .brief-cites { display: flex; flex-direction: column; gap: 4px; }
    .brief-cite {
      display: flex; align-items: baseline; gap: 8px; width: 100%;
      padding: 5px 8px; border-radius: 6px; font-size: 12.5px; text-align: left;
      background: color-mix(in srgb, currentColor 4%, transparent);
      border: 1px solid color-mix(in srgb, currentColor 10%, transparent);
      color: var(--text-primary, #e8eaed);
    }
    button.brief-cite { cursor: pointer; font: inherit; }
    button.brief-cite:hover {
      background: color-mix(in srgb, currentColor 9%, transparent);
      border-color: color-mix(in srgb, currentColor 22%, transparent);
    }
    .brief-cite-icon { flex: 0 0 auto; opacity: 0.8; }
    .brief-cite-label {
      flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
      font-family: var(--font-mono, ui-monospace, monospace);
    }
    .brief-cite-oracle .brief-cite-label { font-family: inherit; }
    .brief-cite-outcome {
      flex: 0 0 auto; font-size: 10.5px; padding: 1px 6px; border-radius: 999px;
      text-transform: uppercase; letter-spacing: 0.3px;
    }
    .brief-cite-green { color: #3fb950; background: color-mix(in srgb, #3fb950 16%, transparent); }
    .brief-cite-red { color: #f85149; background: color-mix(in srgb, #f85149 16%, transparent); }
    .coder-review-file.brief-cite-flash {
      animation: brief-cite-flash 1.2s ease-out;
    }
    @keyframes brief-cite-flash {
      0% { background: color-mix(in srgb, var(--accent, #58a6ff) 30%, transparent); }
      100% { background: transparent; }
    }
  `;
  document.head.appendChild(style);
}

function _short(s, n) {
  s = String(s || '');
  return s.length <= n ? s : s.slice(0, n - 1) + '…';
}

/** Honest verdict line, driven by the independent cross-model verifier
 *  (coder/run_verifier.py). We never imply "done + verified" unless a
 *  different model actually reviewed the diff — and even then the ceiling is
 *  "probable" (a stronger model judged it correct), never "mechanically
 *  proven", because the tests aren't re-run yet. */
function _verdictNote(env) {
  if (!env.ok) {
    const why = env.failure === 'timeout' ? 'timed out'
      : env.failure === 'error' ? 'failed'
      : 'did not complete';
    return `This run ${why}. Any partial work is in the workspace — review the diff below.`;
  }
  const v = env.verification || {};
  const reason = String(v.reason || '').trim();
  const vm = v.verifier_model ? ` (${v.verifier_model})` : '';
  const unmet = Array.isArray(v.contract_unmet) ? v.contract_unmet : [];
  const unmetNote = unmet.length
    ? ` Unmet: ${unmet.slice(0, 3).map((u) => String(u)).join('; ')}${unmet.length > 3 ? '…' : ''}.`
    : '';
  switch (v.tier) {
    case 'verified':
      return `Tests were re-run and passed, and an independent model${vm} agrees. ${reason}`.trim();
    case 'probable':
      return `A different model${vm} reviewed the diff and judges it correct — not mechanically proven (tests weren't run). ${reason}`.trim();
    case 'failed':
      return `An independent check${vm} flagged a problem: ${reason || 'the diff may not satisfy the request.'}${unmetNote} Review the diff before accepting.`;
    case 'human_required':
      return `This needs your decision${vm}: ${reason || 'the request contained an ambiguous choice.'}${unmetNote}`;
    default: // unchecked (no heavyweight pinned, disabled, self, or verifier error)
      return reason
        ? `Not independently verified — ${reason} Review the diff before accepting.`
        : 'Reported complete — not independently verified. Review the diff before accepting.';
  }
}

/** Short badge suffix reflecting the verdict tier. */
function _verdictBadge(env) {
  if (!env.ok) return env.failure || 'failed';
  switch ((env.verification || {}).tier) {
    case 'verified': return 'verified';
    case 'probable': return 'reviewed';
    case 'failed': return 'needs review';
    case 'human_required': return 'your call';
    default: return 'done';
  }
}

async function _fetchPreviewUrl(workspaceId) {
  try {
    const r = await fetch(
      `/api/coder/workspaces/${encodeURIComponent(workspaceId)}/ports`,
      { credentials: 'same-origin' },
    );
    if (!r.ok) return null;
    const d = await r.json();
    const p = d.preview || {};
    if (p.ready) return p.primary_url || (p.urls || [])[0] || null;
    return null;
  } catch (_) {
    return null;
  }
}

async function _mountBriefContent({ hero, body, env }) {
  // ── Hero: live dev-server preview if the workspace has one ready ───────
  if (hero && env.workspace_id) {
    const url = await _fetchPreviewUrl(env.workspace_id);
    if (url) {
      const frame = document.createElement('iframe');
      frame.className = 'brief-hero-frame';
      frame.src = url;
      frame.loading = 'lazy';
      // Sandbox: let the preview run but keep it contained.
      frame.setAttribute('sandbox', 'allow-scripts allow-same-origin allow-forms allow-popups');
      hero.appendChild(frame);
    } else {
      const ph = document.createElement('div');
      ph.className = 'brief-hero-idle';
      ph.textContent = env.ok
        ? 'No live preview — the workspace has no running dev server.'
        : 'Run did not complete.';
      hero.appendChild(ph);
    }
  }

  // ── Body: the code diff (Accept/Reject per-file lives inside the card) ─
  if (body) {
    const section = document.createElement('div');
    section.className = 'brief-section';
    const h = document.createElement('div');
    h.className = 'brief-section-title';
    h.textContent = 'Changes';
    section.appendChild(h);
    const diffHost = document.createElement('div');
    section.appendChild(diffHost);
    body.appendChild(section);

    let diffCard = null;
    if (env.review_turn_id) {
      try {
        const { mountReviewPanel } = await import('./coder-review.js');
        diffCard = await mountReviewPanel(env.review_turn_id, diffHost);
        if (!diffCard) diffHost.innerHTML = '<div class="brief-empty">No reviewable changes for this run.</div>';
      } catch (err) {
        console.warn('[brief] diff mount failed', err);
        diffHost.innerHTML = '<div class="brief-empty">Diff unavailable.</div>';
      }
    } else {
      diffHost.innerHTML = '<div class="brief-empty">No file changes recorded for this run.</div>';
    }

    // ── Evidence: the citation ledger, an index INTO the diff above ──────
    // The diff IS the transparency surface (the user sees the exact changed
    // code before deciding); citations let them jump from a specific claim
    // ("added the parser") to the exact file+lines that back it, and show
    // which oracles actually ran (tests-not-gamed).
    if (env.review_turn_id) {
      await _mountCitations(body, env.review_turn_id, diffCard);
    }
  }
}

/** Fetch + render the citation ledger for a turn as a compact evidence list.
 *  Write citations deep-link into the mounted diff card; oracle citations
 *  show what was actually verified (green/red). Silent + honest when empty. */
async function _mountCitations(body, turnRunId, diffCard) {
  let rows = [];
  try {
    const r = await fetch(
      `/api/coder/runs/${encodeURIComponent(turnRunId)}/citations`,
      { credentials: 'same-origin' },
    );
    if (r.ok) rows = (await r.json()).citations || [];
  } catch (_) { /* evidence is best-effort — absence is a valid state */ }
  if (!rows.length) return;  // no citations → the diff already stands alone

  const section = document.createElement('div');
  section.className = 'brief-section';
  const h = document.createElement('div');
  h.className = 'brief-section-title';
  h.textContent = 'Evidence';
  section.appendChild(h);

  const list = document.createElement('div');
  list.className = 'brief-cites';
  for (const c of rows) {
    list.appendChild(_citationRow(c, diffCard));
  }
  section.appendChild(list);
  body.appendChild(section);
}

/** One citation row. Write kinds are a button that reveals the backing file
 *  in the diff; oracle kinds render an outcome chip. */
function _citationRow(c, diffCard) {
  const kind = String(c.evidence_kind || '');
  const isWrite = kind === 'write';
  const span = (c.line_start && c.line_end) ? `:${c.line_start}-${c.line_end}` : '';
  const label = isWrite
    ? `${escapeHtml(c.file || '(file)')}${span}`
    : `${escapeHtml(kind)} — ${escapeHtml(c.evidence_ref || '')}`;

  const row = document.createElement(isWrite && c.file ? 'button' : 'div');
  row.className = `brief-cite brief-cite-${isWrite ? 'write' : 'oracle'}`;
  const icon = isWrite ? '✎' : (c.outcome === 'green' ? '✓' : c.outcome === 'red' ? '✗' : '•');
  const chip = (!isWrite && c.outcome)
    ? `<span class="brief-cite-outcome brief-cite-${escapeHtml(c.outcome)}">${escapeHtml(c.outcome)}</span>`
    : '';
  row.innerHTML = `<span class="brief-cite-icon">${icon}</span>`
    + `<span class="brief-cite-label">${label}</span>${chip}`;

  if (isWrite && c.file) {
    row.type = 'button';
    row.addEventListener('click', () => _revealFileInDiff(diffCard, c.file));
  }
  return row;
}

/** Scroll to + expand the changed file inside the mounted review card, so a
 *  citation drills straight into the exact code the user is judging. */
function _revealFileInDiff(diffCard, file) {
  if (!diffCard || !file) return;
  // Ensure the review body is open (it starts collapsed).
  const cardBody = diffCard.querySelector('.coder-review-body');
  if (cardBody && cardBody.hidden) {
    const header = diffCard.querySelector('.coder-review-header');
    header?.click();
  }
  const fileRow = diffCard.querySelector(`.coder-review-file[data-path="${CSS.escape(file)}"]`);
  if (!fileRow) return;
  // Expand this file's diff (click its head if the pre is still hidden).
  const pre = fileRow.querySelector('.coder-review-diff');
  const head = fileRow.querySelector('.coder-review-file-head');
  if (pre && pre.hidden && head) head.click();
  fileRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
  fileRow.classList.add('brief-cite-flash');
  setTimeout(() => fileRow.classList.remove('brief-cite-flash'), 1200);
}

/** Fetch the persisted verdict for a run — the cold-open fallback when the
 *  brief is reopened past its envelope (e.g. from a stale notification). */
async function _fetchVerification(runId) {
  try {
    const r = await fetch(
      `/api/coder/runs/${encodeURIComponent(runId)}/verification`,
      { credentials: 'same-origin' },
    );
    if (!r.ok) return null;
    const d = await r.json();
    return d.verification || null;
  } catch (_) {
    return null;
  }
}

/** Open (or replace) the companion brief for a completed coder run. */
export async function openBrief(envelope) {
  const env = envelope || {};
  const wsId = env.workspace_id || '';
  _ensureStyles();

  // Live opens carry the verdict in the envelope; a cold reopen (stale
  // notification deep-link) does not — fetch it so the verdict is never
  // silently missing.
  if (!env.verification && env.run_id) {
    const v = await _fetchVerification(env.run_id);
    if (v) env.verification = v;
  }

  const actions = [];
  if (wsId) {
    actions.push({
      label: 'Open in Coder',
      primary: true,
      onClick: () => {
        try { window.openCoderWorkspace?.(wsId); } catch (_) { /* nav best-effort */ }
        closeCompanionDetail();
      },
    });
    // Save to Library — reuse coder-library-save's popover. Best-effort;
    // some run kinds aren't library-savable, so failures are silent.
    actions.push({
      label: 'Save to Library',
      onClick: async () => {
        try {
          const { openSavePrompt } = await import('./coder-library-save.js');
          const anchor = document.querySelector('#companion-detail-panel .cd-actions') || document.body;
          await openSavePrompt(wsId, anchor);
        } catch (err) {
          console.warn('[brief] save-to-library failed', err);
        }
      },
    });
  }

  openCompanionDetail({
    title: env.prompt ? _short(env.prompt, 90) : 'Coder run',
    subtitle: env.answer_preview ? _short(env.answer_preview, 120) : '',
    badge: `Coder · ${_verdictBadge(env)}`,
    note: _verdictNote(env),
    fields: [
      env.tool_calls ? { label: 'tools', value: String(env.tool_calls) } : null,
      env.elapsed_s ? { label: 'took', value: `${env.elapsed_s}s` } : null,
      wsId ? { label: 'workspace', value: _short(wsId, 14) } : null,
    ].filter(Boolean),
    actions,
    mount: ({ hero, body }) => { _mountBriefContent({ hero, body, env }); },
  });
}

// ── Global + dev hook ────────────────────────────────────────────────────
// openCompanionBrief is the integration point the surface-event router (and,
// later, the presence-bus consumer) calls when the backend emits a
// coder_run_completed / companion.brief_open signal. __previewBrief lets you
// render the panel now, before the backend changes are loaded.
if (typeof window !== 'undefined') {
  window.openCompanionBrief = openBrief;
  window.__previewBrief = (env) => openBrief(env || {
    ok: true,
    kind: 'coder_run',
    workspace_id: window.__briefDemoWs || '',
    run_id: 'demo-run',
    review_turn_id: window.__briefDemoTurn || '',
    prompt: 'Add a dark-mode toggle to the settings page',
    answer_preview: 'Added a theme toggle wired to localStorage and the CSS variables; tests pass.',
    tool_calls: 12,
    elapsed_s: 143,
  });
}
