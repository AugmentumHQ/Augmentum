/**
 * Reviewable-turn panel for coder mode.
 *
 * Flow (see augmentum/coder/reviews.py for the backend contract):
 *
 *  1. The coder agent finishes a turn. phase_act.py's final meta
 *     chunk carries `aug.status == 'complete'` plus
 *     `aug.review_turn_id` — a stable handle for the bundle that
 *     ReviewRegistry just published.
 *  2. coder-stream fires `onReviewPending(turnId)`. The conversation
 *     layer calls `mountReviewPanel(turnId, container)` (us), which
 *     fetches the bundle and renders an inline review card.
 *  3. User clicks Accept / Reject / Partial. We POST to the
 *     corresponding endpoint. The response reports what happened
 *     (restored paths, commit hash, any failed restores). We update
 *     the card in-place so the user sees confirmation, then fire a
 *     `coder:turn-reviewed` event so the file tree and conversation
 *     can refresh.
 *
 * Default state is COLLAPSED. The turn ended with a synthesis
 * message the user has just read; surfacing diffs without asking
 * creates scroll noise. Prominent header with file count + quick
 * Accept/Reject buttons keeps the lightweight path one click away;
 * clicking the header (or a file name) expands for review.
 *
 * Scope — Sprint 1:
 *   • Turn-level Accept / Reject. Per-file Partial with checkboxes.
 *   • Unified-diff rendering with status-coloured gutters.
 *   • Non-reversible paths flagged so a reject doesn't surprise.
 * Deferred to Sprint 2:
 *   • Per-hunk granularity (parsing unified diffs into hunks).
 *   • Rejection-reason capture → next-turn context.
 *   • Keyboard navigation.
 *   • Diff from binary-changed files (currently rendered as-is).
 */
import { escapeHtml } from './app.js';


// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Fetch a turn's review bundle and mount the panel into `container`.
 *
 * @param {string} turnId        — the bundle's turn_id.
 * @param {HTMLElement} container — where to append the card. Usually
 *                                  the coder-conv-messages scroll
 *                                  element, so the panel lives in the
 *                                  chat flow right after the synthesis
 *                                  message.
 * @returns {Promise<HTMLElement|null>}  the mounted card, or null on
 *                                       fetch failure (backend off,
 *                                       turn unknown, cross-tenant).
 */
export async function mountReviewPanel(turnId, container) {
  if (!turnId || !container) return null;

  let bundle;
  try {
    const resp = await fetch(
      `/api/coder/reviews/${encodeURIComponent(turnId)}`,
    );
    if (!resp.ok) {
      // 404 is normal — the bundle may have already been resolved by
      // another client or rejected by the server (e.g. reviews disabled).
      // Silent on non-200; the user doesn't need a toast for this.
      return null;
    }
    bundle = await resp.json();
  } catch {
    return null;
  }

  if (!bundle || !Array.isArray(bundle.files) || bundle.files.length === 0) {
    // Zero-diff turns don't get a panel. The backend's
    // _publish_turn_review already skips them, so this is a defensive
    // guard against future API changes.
    return null;
  }

  const card = _renderCard(bundle);
  container.appendChild(card);

  // Keep the just-mounted card in view. Without this, the appended
  // card can scroll off-screen on a turn that ended with a long
  // synthesis message.
  requestAnimationFrame(() => {
    container.scrollTop = container.scrollHeight;
  });

  return card;
}


// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/**
 * Render the full review card (collapsed by default).
 * DOM shape is cheap enough we build it all upfront and toggle
 * visibility via classes rather than lazy-rendering the diffs. 20+
 * files is rare; even then ~50KB of DOM is negligible.
 */
function _renderCard(bundle) {
  const card = document.createElement('div');
  card.className = 'coder-review-card';
  card.dataset.turnId = bundle.turn_id;
  card.dataset.state = 'pending';

  const s = bundle.summary;
  const summaryText = _summaryText(s);
  const nonReversibleFlag = s.non_reversible > 0
    ? `<span class="coder-review-nonrev" title="${s.non_reversible} path(s) can't be fully undone — see file list">
         ${s.non_reversible} non-reversible
       </span>`
    : '';

  // Header: click toggles expand. Accept/Reject buttons live here so
  // the happy path is one click from the collapsed state.
  card.innerHTML = `
    <button class="coder-review-header" type="button">
      <svg class="coder-review-chevron" viewBox="0 0 24 24" width="14" height="14"
           fill="none" stroke="currentColor" stroke-width="2"
           stroke-linecap="round" stroke-linejoin="round">
        <polyline points="9 18 15 12 9 6"/>
      </svg>
      <span class="coder-review-title">Review</span>
      <span class="coder-review-summary">${escapeHtml(summaryText)}</span>
      ${nonReversibleFlag}
      <span class="coder-review-spacer"></span>
      <span class="coder-review-actions">
        <button type="button" class="coder-review-btn coder-review-reject"
                title="Revert every file to its pre-turn state">Reject</button>
        <button type="button" class="coder-review-btn coder-review-accept"
                title="Keep all changes + commit">Accept</button>
      </span>
    </button>
    <div class="coder-review-body" hidden>
      <div class="coder-review-files"></div>
      <div class="coder-review-partial-footer" hidden>
        <span class="coder-review-partial-hint">Checked = keep · Unchecked = revert</span>
        <button type="button" class="coder-review-btn coder-review-apply">Apply selection</button>
      </div>
      <div class="coder-review-outcome" hidden></div>
    </div>
  `;

  const filesEl = card.querySelector('.coder-review-files');
  for (const f of bundle.files) {
    filesEl.appendChild(_renderFileRow(f));
  }

  _wireInteractions(card, bundle);
  return card;
}


/**
 * One file's row — checkbox + status badge + path + size delta.
 * The diff itself is rendered into an inline `<pre>` that toggles
 * visibility when the user clicks the row (distinct from the
 * checkbox click, which toggles keep/revert).
 */
function _renderFileRow(file) {
  const row = document.createElement('div');
  row.className = 'coder-review-file';
  row.dataset.path = file.path;
  row.dataset.reversible = file.reversible ? '1' : '0';

  const statusBadge = _statusBadge(file.status);
  const reversibilityHint = file.reversible
    ? ''
    : `<span class="coder-review-file-flag"
              title="Pre-turn state wasn't captured — reject won't fully undo">
         non-reversible
       </span>`;
  const delta = _sizeDelta(file.old_size, file.new_size);

  row.innerHTML = `
    <label class="coder-review-file-head">
      <input type="checkbox" class="coder-review-keep" checked
             title="Checked = keep this file's changes">
      <span class="coder-review-file-status">${statusBadge}</span>
      <span class="coder-review-file-path">${escapeHtml(file.path)}</span>
      <span class="coder-review-file-delta">${delta}</span>
      ${reversibilityHint}
      <svg class="coder-review-file-chevron" viewBox="0 0 24 24" width="12" height="12"
           fill="none" stroke="currentColor" stroke-width="2"
           stroke-linecap="round" stroke-linejoin="round">
        <polyline points="9 18 15 12 9 6"/>
      </svg>
    </label>
    <pre class="coder-review-diff" hidden></pre>
  `;

  // Populate the <pre> with a formatted diff. Done after setting
  // innerHTML so we can use the safer node-based insertion that
  // preserves per-line class decoration.
  const pre = row.querySelector('.coder-review-diff');
  pre.appendChild(_renderUnifiedDiff(file.unified_diff));
  return row;
}


/**
 * Convert a unified-diff string into a DocumentFragment with one
 * class-tagged `<span>` per line for gutter colouring.
 * Conservative parser — handles the output of python difflib's
 * unified_diff (which is what turn_snapshot.py emits). Lines are
 * classified by leading char:
 *
 *   '--- ' / '+++ ' → file header
 *   '@@ '           → hunk header
 *   '+'             → added
 *   '-'             → removed
 *   ' '             → context
 *   else            → rendered plain (e.g. "\ No newline at EOF")
 */
function _renderUnifiedDiff(text) {
  const frag = document.createDocumentFragment();
  if (!text) {
    const empty = document.createElement('span');
    empty.className = 'coder-review-diff-line coder-review-diff-empty';
    empty.textContent = '(no textual diff — file may be binary or unchanged)';
    frag.appendChild(empty);
    return frag;
  }
  const lines = text.split('\n');
  // difflib emits a trailing newline → empty last element; drop it
  // so we don't render a spurious blank row.
  if (lines.length && lines[lines.length - 1] === '') lines.pop();

  for (const line of lines) {
    const el = document.createElement('span');
    el.className = 'coder-review-diff-line';
    if (line.startsWith('--- ') || line.startsWith('+++ ')) {
      el.classList.add('coder-review-diff-header');
    } else if (line.startsWith('@@')) {
      el.classList.add('coder-review-diff-hunk');
    } else if (line.startsWith('+')) {
      el.classList.add('coder-review-diff-add');
    } else if (line.startsWith('-')) {
      el.classList.add('coder-review-diff-del');
    } else {
      el.classList.add('coder-review-diff-ctx');
    }
    el.textContent = line + '\n';
    frag.appendChild(el);
  }
  return frag;
}


function _summaryText(s) {
  // Compact one-line summary. "3 files changed · 2 added · 1 modified"
  // reads cleaner than "3 files changed (added: 2, modified: 1)".
  const parts = [`${s.files_changed} file${s.files_changed === 1 ? '' : 's'} changed`];
  if (s.added)    parts.push(`${s.added} added`);
  if (s.modified) parts.push(`${s.modified} modified`);
  if (s.deleted)  parts.push(`${s.deleted} deleted`);
  return parts.join(' · ');
}

function _statusBadge(status) {
  // Symbolic prefix rather than text — saves horizontal space and
  // mirrors git's convention. Classes drive colour in CSS.
  const map = {
    added:    ['+', 'add'],
    modified: ['~', 'mod'],
    deleted:  ['-', 'del'],
  };
  const [sym, cls] = map[status] || ['?', 'unknown'];
  return `<span class="coder-review-badge coder-review-badge-${cls}">${sym}</span>`;
}

function _sizeDelta(oldSize, newSize) {
  // Show the byte change as "+N / -N" the way git does with lines.
  // Operate on bytes rather than lines because we'd have to re-parse
  // the unified diff to know line counts — not worth it for a hint.
  if (oldSize === 0 && newSize > 0) return `<span class="coder-review-delta-add">+${newSize}</span>`;
  if (newSize === 0 && oldSize > 0) return `<span class="coder-review-delta-del">−${oldSize}</span>`;
  const delta = newSize - oldSize;
  const sign = delta >= 0 ? '+' : '−';
  const cls = delta >= 0 ? 'coder-review-delta-add' : 'coder-review-delta-del';
  return `<span class="${cls}">${sign}${Math.abs(delta)} bytes</span>`;
}


// ---------------------------------------------------------------------------
// Interactions
// ---------------------------------------------------------------------------

function _wireInteractions(card, bundle) {
  const header = card.querySelector('.coder-review-header');
  const body = card.querySelector('.coder-review-body');
  const filesEl = card.querySelector('.coder-review-files');
  const partialFooter = card.querySelector('.coder-review-partial-footer');
  const outcomeEl = card.querySelector('.coder-review-outcome');
  const actionsEl = card.querySelector('.coder-review-actions');

  const acceptBtn = card.querySelector('.coder-review-accept');
  const rejectBtn = card.querySelector('.coder-review-reject');
  const applyBtn = card.querySelector('.coder-review-apply');

  // Header click toggles expand. Buttons inside the header stop
  // propagation so clicking Accept doesn't also expand.
  header.addEventListener('click', (e) => {
    if (e.target.closest('.coder-review-actions')) return;
    const expanded = card.classList.toggle('is-expanded');
    body.hidden = !expanded;
    card.querySelector('.coder-review-chevron')
      ?.style.setProperty('transform', expanded ? 'rotate(90deg)' : 'rotate(0)');
  });

  // Click a file row head to toggle its diff open.
  filesEl.addEventListener('click', (e) => {
    // Checkbox clicks must NOT toggle the diff — checkbox changes
    // whether that file is kept on Apply. Let the label's native
    // checkbox behaviour handle the check; only respond to clicks
    // that weren't on the input.
    if (e.target.matches('input[type="checkbox"]')) return;
    // If the click came from inside a label that contains a checkbox,
    // the browser ALSO fires a click on the checkbox. We want the
    // diff-toggle only for label clicks where the actual target is
    // not the input — so bail when the target is specifically the
    // checkbox (covered above) or a descendent of it (not possible
    // here since the input is a leaf). Good.
    const row = e.target.closest('.coder-review-file');
    if (!row) return;

    const pre = row.querySelector('.coder-review-diff');
    const chevron = row.querySelector('.coder-review-file-chevron');
    pre.hidden = !pre.hidden;
    if (chevron) {
      chevron.style.transform = pre.hidden ? 'rotate(0)' : 'rotate(90deg)';
    }
  });

  // Show the partial footer as soon as any checkbox changes state,
  // so the user sees the "Apply selection" path light up without
  // having to hunt for a mode toggle.
  filesEl.addEventListener('change', (e) => {
    if (!e.target.matches('input.coder-review-keep')) return;
    const anyUnchecked = Array.from(
      filesEl.querySelectorAll('input.coder-review-keep'),
    ).some(i => !i.checked);
    partialFooter.hidden = !anyUnchecked;
    // When a checkbox is touched, the turn-level Accept/Reject
    // buttons stop matching user intent — they'd override the
    // per-file selection. Dim them to signal Apply-selection is the
    // action that will actually use the checkboxes.
    actionsEl.classList.toggle('is-dimmed', anyUnchecked);
  });

  acceptBtn.addEventListener('click', async () => {
    _setCardState(card, 'working');
    const res = await _post(`/api/coder/reviews/${bundle.turn_id}/accept`);
    _finalise(card, res, outcomeEl, 'accepted');
  });

  rejectBtn.addEventListener('click', async () => {
    if (!confirm(
      `Reject this turn? Every ${bundle.files.length} file change will be rolled back.`,
    )) return;
    _setCardState(card, 'working');
    const res = await _post(`/api/coder/reviews/${bundle.turn_id}/reject`);
    _finalise(card, res, outcomeEl, 'rejected');
  });

  applyBtn.addEventListener('click', async () => {
    const accepted = [];
    const rejected = [];
    filesEl.querySelectorAll('.coder-review-file').forEach(row => {
      const keep = row.querySelector('input.coder-review-keep').checked;
      (keep ? accepted : rejected).push(row.dataset.path);
    });
    _setCardState(card, 'working');
    const res = await _post(`/api/coder/reviews/${bundle.turn_id}/partial`, {
      accepted_paths: accepted,
      rejected_paths: rejected,
    });
    _finalise(card, res, outcomeEl, 'partial');
  });
}

function _setCardState(card, state) {
  card.dataset.state = state;
  // Disable the buttons while the backend is working — double-click
  // protection AND visual signal that something's happening.
  card.querySelectorAll('.coder-review-btn').forEach(b => {
    b.disabled = (state === 'working' || state === 'done');
  });
}

function _finalise(card, res, outcomeEl, fallbackStatus) {
  if (!res || res.error) {
    _setCardState(card, 'pending');
    outcomeEl.hidden = false;
    outcomeEl.className = 'coder-review-outcome is-error';
    outcomeEl.textContent = `Action failed: ${res?.error || 'network error'}. Try again.`;
    return;
  }

  _setCardState(card, 'done');
  outcomeEl.hidden = false;

  const status = res.status || fallbackStatus;
  const lines = [];
  if (status === 'accepted') {
    lines.push('Accepted.');
    if (res.commit) lines.push(`Committed as ${res.commit}.`);
  } else if (status === 'rejected') {
    lines.push(`Rejected — ${res.restored_paths?.length ?? 0} file(s) restored.`);
    if (res.failed_paths?.length) {
      lines.push(
        `${res.failed_paths.length} file(s) could not be auto-restored: ${res.failed_paths.join(', ')}.`,
      );
    }
  } else if (status === 'partial') {
    lines.push(`Applied selection — kept ${res.accepted_paths?.length ?? 0}, reverted ${res.rejected_paths?.length ?? 0}.`);
    if (res.commit) lines.push(`Committed kept files as ${res.commit}.`);
    if (res.failed_paths?.length) {
      lines.push(`Failed to revert: ${res.failed_paths.join(', ')}.`);
    }
  }
  outcomeEl.className = 'coder-review-outcome is-done';
  outcomeEl.textContent = lines.join(' ');

  // Let the rest of the app know the turn was reviewed — file tree
  // should refresh (some files may have been rolled back), and the
  // conversation layer can clear any "pending review" hint.
  document.dispatchEvent(new CustomEvent('coder:turn-reviewed', {
    detail: {
      turnId: card.dataset.turnId,
      status,
      result: res,
    },
  }));
}


// ---------------------------------------------------------------------------
// HTTP helper
// ---------------------------------------------------------------------------

async function _post(url, body) {
  try {
    const resp = await fetch(url, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    body ? JSON.stringify(body) : undefined,
    });
    // 4xx / 5xx still return JSON (see coder_review_routes); surface
    // the `error` field rather than falling through to a generic
    // message.
    const data = await resp.json().catch(() => null);
    if (!resp.ok) return { error: data?.error || `HTTP ${resp.status}` };
    return data || { error: 'empty response' };
  } catch (err) {
    return { error: err.message || 'network error' };
  }
}
