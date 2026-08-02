import { escapeHtml } from '../app.js';

function _domainLabel(url = '') {
  try {
    return (new URL(url)).hostname.replace(/^www\./, '');
  } catch {
    return '';
  }
}

function _renderSourceRows(items = [], limit = 8) {
  return items.slice(0, limit).map((item, index) => {
    const rank = item.rank || index + 1;
    const url = item.url || '';
    const title = item.title || _domainLabel(url) || url || `Source ${rank}`;
    const domain = item.source || _domainLabel(url);
    return `
      <a class="tool-source-row" href="${escapeHtml(url)}" target="_blank" rel="noopener">
        <span class="tool-source-rank">${rank}</span>
        <span class="tool-source-main">
          <span class="tool-source-title">${escapeHtml(title)}</span>
          ${domain ? `<span class="tool-source-domain">${escapeHtml(domain)}</span>` : ''}
        </span>
      </a>
    `;
  }).join('');
}

function _dedupeSources(items = []) {
  const seen = new Set();
  const deduped = [];
  for (const item of items) {
    const url = item?.url || '';
    if (!url || seen.has(url)) continue;
    seen.add(url);
    deduped.push(item);
  }
  return deduped;
}

function _renderSources(items = [], summaryLabel = 'Sources') {
  if (!Array.isArray(items) || !items.length) return '';
  return `
    <section class="tool-result-section">
      <details class="tool-result-disclosure">
        <summary>${escapeHtml(summaryLabel)}</summary>
        <div class="tool-source-list">
          ${_renderSourceRows(items)}
        </div>
      </details>
    </section>
  `;
}

function _renderSourceSummary(items = []) {
  if (!Array.isArray(items) || !items.length) return '';
  const deduped = _dedupeSources(items);
  const summary = `${deduped.length} source${deduped.length === 1 ? '' : 's'}`;
  return `
    <section class="tool-result-section">
      <div class="tool-result-summary">
        <span class="tool-result-summary-count">${escapeHtml(summary)}</span>
      </div>
      ${_renderSources(deduped, 'Sources')}
    </section>
  `;
}

function _renderWebFetch(meta = {}) {
  const page = meta.page || {
    url: meta.url || '',
    title: meta.title || '',
    source: _domainLabel(meta.url || ''),
  };
  return _renderSourceSummary([page]);
}

function _renderWebSearch(meta = {}) {
  return _renderSourceSummary(meta.results || []);
}

function _renderWeb(meta = {}) {
  const combined = [];
  if (meta.results?.length) combined.push(...meta.results);
  if (meta.page) combined.push(meta.page);
  if (meta.fetched_pages?.length) combined.push(...meta.fetched_pages);
  return _renderSourceSummary(combined);
}

// ---------------------------------------------------------------------------
// task_dispatch — render a spawned subagent as a structured card under the
// parent turn. Surfaces role, resolved model (often shows fabric peer or
// API provider routing), iterations / tokens / wallclock, stop reason, and
// a click-to-load full transcript link wired to the subagent run-detail
// route. Designed for the cooperative coder + lead-model UX where the
// lead invokes task_dispatch via the OpenAI tool layer.
// ---------------------------------------------------------------------------

function _formatTokens(n) {
  const v = Number(n) || 0;
  if (v < 1000) return String(v);
  if (v < 1_000_000) return `${(v / 1000).toFixed(1)}k`;
  return `${(v / 1_000_000).toFixed(2)}M`;
}

function _formatDuration(ms) {
  const v = Number(ms) || 0;
  if (v < 1000) return `${v}ms`;
  if (v < 60_000) return `${(v / 1000).toFixed(1)}s`;
  const min = Math.floor(v / 60_000);
  const sec = Math.round((v % 60_000) / 1000);
  return `${min}m ${sec}s`;
}

function _stopReasonPill(stop) {
  const s = String(stop || '').toLowerCase();
  const map = {
    complete: { label: 'complete', cls: 'subagent-stop-complete' },
    budget:   { label: 'budget',   cls: 'subagent-stop-budget' },
    stuck:    { label: 'stuck',    cls: 'subagent-stop-stuck' },
    error:    { label: 'error',    cls: 'subagent-stop-error' },
  };
  const entry = map[s] || { label: s || 'unknown', cls: 'subagent-stop-unknown' };
  return `<span class="subagent-stop-pill ${entry.cls}">${escapeHtml(entry.label)}</span>`;
}

function _renderTaskDispatch(meta = {}) {
  const role = String(meta.role || 'subagent');
  const modelResolved = String(meta.model_resolved || '');
  const modelSpec = String(meta.model_spec || '');
  const subagentId = String(meta.subagent_id || '');
  const iterations = Number(meta.iterations) || 0;
  const toolCalls = Number(meta.tool_calls) || 0;
  const tokensIn = Number(meta.tokens_in) || 0;
  const tokensOut = Number(meta.tokens_out) || 0;
  const wallclockMs = Number(meta.wallclock_ms) || 0;
  const stopReason = String(meta.stop_reason || '');
  const stopDetail = String(meta.stop_detail || '');
  const stuckPattern = String(meta.stuck_pattern || '');

  const modelLabel = modelSpec && modelSpec !== modelResolved
    ? `${modelResolved} (${modelSpec})`
    : modelResolved;

  const detailLine = stopDetail
    ? `<div class="subagent-stop-detail">${escapeHtml(stopDetail)}</div>`
    : '';
  const stuckLine = stuckPattern
    ? `<div class="subagent-stop-detail">stuck pattern: ${escapeHtml(stuckPattern)}</div>`
    : '';

  const inspectLink = subagentId
    ? `<a class="subagent-inspect-link" href="/api/coder/subagents/${encodeURIComponent(subagentId)}" target="_blank" rel="noopener" title="Full transcript + tool-call log">inspect →</a>`
    : '';

  return `
    <section class="tool-result-section subagent-card">
      <div class="subagent-card-header">
        <div class="subagent-card-title">
          <span class="subagent-role-tag">${escapeHtml(role)}</span>
          <span class="subagent-model">${escapeHtml(modelLabel || '(default model)')}</span>
        </div>
        ${_stopReasonPill(stopReason)}
      </div>
      <dl class="subagent-stats">
        <div class="subagent-stat">
          <dt>iters</dt>
          <dd>${iterations}</dd>
        </div>
        <div class="subagent-stat">
          <dt>tools</dt>
          <dd>${toolCalls}</dd>
        </div>
        <div class="subagent-stat">
          <dt>tokens</dt>
          <dd>${_formatTokens(tokensIn + tokensOut)} <span class="subagent-stat-sub">(${_formatTokens(tokensIn)} in / ${_formatTokens(tokensOut)} out)</span></dd>
        </div>
        <div class="subagent-stat">
          <dt>wall</dt>
          <dd>${_formatDuration(wallclockMs)}</dd>
        </div>
      </dl>
      ${detailLine}
      ${stuckLine}
      ${inspectLink ? `<div class="subagent-card-footer">${inspectLink}</div>` : ''}
    </section>
  `;
}

/* Code execution — the source is shown, collapsed behind a disclosure arrow.
   Code that ran on the user's own machine must be reviewable after the fact,
   so this renders from persisted `result_metadata.code` rather than the
   transient 120-char tool_start subtitle. Keyed off the presence of `code`
   (not the tool name) so any future executor tool gets the same treatment
   by returning `code` in its ToolResult.metadata. */
function _renderCodeExec(meta = {}) {
  const code = typeof meta.code === 'string' ? meta.code : '';
  if (!code.trim()) return '';

  const lang = escapeHtml(meta.language || 'code');
  const lines = code.split('\n').length;
  const elapsed = Number(meta.elapsed_seconds);
  const timing = Number.isFinite(elapsed) ? `${elapsed.toFixed(2)}s` : '';
  const stdout = typeof meta.stdout === 'string' ? meta.stdout : '';
  const stderr = typeof meta.stderr === 'string' ? meta.stderr : '';

  const streams = [
    stdout.trim() ? ['stdout', stdout, ''] : null,
    stderr.trim() ? ['stderr', stderr, ' tool-code-stream--err'] : null,
  ].filter(Boolean).map(([label, body, mod]) => `
    <div class="tool-code-stream${mod}">
      <span class="tool-code-stream-label">${label}</span>
      <pre class="tool-code-block"><code>${escapeHtml(body)}</code></pre>
    </div>
  `).join('');

  return `
    <section class="tool-result-section">
      <details class="tool-result-disclosure tool-code-disclosure">
        <summary>
          <span class="tool-code-summary-main">
            <span class="tool-code-caret" aria-hidden="true"></span>
            <span class="tool-code-lang">${lang}</span>
            <span class="tool-code-meta">${lines} line${lines === 1 ? '' : 's'}${timing ? ` · ${escapeHtml(timing)}` : ''}</span>
          </span>
        </summary>
        <pre class="tool-code-block tool-code-block--source"><code>${escapeHtml(code)}</code></pre>
        ${streams}
      </details>
    </section>
  `;
}

export function renderToolResultView(toolName, meta = {}) {
  if (!meta || typeof meta !== 'object') return '';
  if (typeof meta.code === 'string' && meta.code.trim()) {
    return _renderCodeExec(meta);
  }
  switch (toolName) {
    case 'web_search':
      return _renderWebSearch(meta);
    case 'web_fetch':
      return _renderWebFetch(meta);
    case 'web':
      return _renderWeb(meta);
    case 'task_dispatch':
      return _renderTaskDispatch(meta);
    default:
      return '';
  }
}
