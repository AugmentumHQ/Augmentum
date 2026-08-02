/* ==========================================================================
   Chat Module — MessageRenderer
   Per-instance DOM creation, message rendering, streaming, scroll tracking.
   Enables multiple independent ChatSurface instances — no singletons.
   ========================================================================== */

import { escapeHtml } from '../app.js';
import { stripMotionCueStreaming } from '../motion-cue.js';
import { icons, PHASE_DISPLAY_NAMES } from './constants.js';
import { renderMarkdown, highlightCode, highlightCodeDeferred, addLineNumbers } from './markdown.js';
import { renderToolResultView } from './tool-result-view.js';
import { renderStreamSplit, newSplitState } from './stream-render.js';
import { planThinkingAppend } from './thinking-stream.js';
import { getActivePath, getSiblingInfo, sessionHasMessages } from './tree.js';
import { copyToClipboard } from '../clipboard.js';
import { bus } from '../activity-bus.js';

// ---------------------------------------------------------------------------
// Helpers (private to module)
// ---------------------------------------------------------------------------

// Lazy-loaded cache for the avatar module. Three states:
//   null  — never tried to load yet
//   false — load failed once, do not retry (avatar not available on this surface)
//   <mod> — module resolved, ready to call
//
// Cached at module scope so all MessageRenderer instances share the same
// reference + the same one-shot load. Without this we'd allocate a
// promise per streaming chunk just to discover the same answer; with it
// the per-delta path is a single synchronous null-check.
let _avatarMod = null;
let _avatarModLoading = false;

/** Forward a streaming text chunk to the avatar presence engine.
 *
 *  Voice mode already does this via voice.js — this helper closes the
 *  chat-only gap so the VRMA picker + presence engine stay in sync with
 *  what's being streamed in text mode. The avatar module decides
 *  internally whether anything is mounted; calling onLLMDelta when no
 *  avatar is up is a cheap no-op there.
 *
 *  Synchronous, allocation-light: the first call kicks off a one-shot
 *  dynamic import; subsequent calls hit the cached module directly.
 *  Failure once → never retry (`_avatarMod = false`).
 */
function _avatarOnDelta(text) {
  if (!text) return;
  if (_avatarMod) {
    try { _avatarMod.onLLMDelta(text); }
    catch (err) {
      // One-shot warn then suppress — avoid spamming the console on
      // every chunk if the presence engine throws repeatedly.
      if (!_avatarMod._chatOnLLMDeltaErrorLogged) {
        console.warn('[chat] avatar.onLLMDelta threw — disabling forwarding for this surface', err);
        _avatarMod._chatOnLLMDeltaErrorLogged = true;
      }
    }
    return;
  }
  if (_avatarMod === false || _avatarModLoading) return;
  _avatarModLoading = true;
  import('../avatar.js')
    .then((m) => {
      if (m && typeof m.onLLMDelta === 'function') _avatarMod = m;
      else _avatarMod = false;  // module loaded but no onLLMDelta export — give up
    })
    .catch(() => { _avatarMod = false; })
    .finally(() => { _avatarModLoading = false; });
}

const _VALID_PHASE_STATUSES = new Set(['complete', 'running', 'pending', 'skipped']);
function _safeStatus(s) { return _VALID_PHASE_STATUSES.has(s) ? s : 'pending'; }

/** Friendly status labels for the streaming progress indicator. */
const _STATUS_LABELS = {
  thinking:     'Thinking',
  composing:    'Composing response',
  planning:     'Planning approach',
  synthesizing: 'Synthesizing results',
  loading:      'Loading model\u2026',
  swapping:     'Switching model\u2026',
  restoring:    'Restoring session\u2026',
  tokenizing:   'Preparing context\u2026',
};

/** Action-oriented labels shown in the streaming dots while a tool is running. */
const _TOOL_STATUS_LABELS = {
  web: 'Searching the web', web_search: 'Searching the web',
  web_fetch: 'Fetching page', wikipedia: 'Checking Wikipedia',
  youtube_transcript: 'Looking up video', calculator: 'Calculating',
  datetime: 'Checking the time', unit_converter: 'Converting units',
  image_generation: 'Generating image', python_exec: 'Running code',
  document_parse: 'Reading document', memory_recall: 'Checking memory',
  create_document: 'Creating document', create_presentation: 'Building slides',
  create_spreadsheet: 'Building spreadsheet', create_chart: 'Creating chart',
};

// Narrative internal tools (memory recall + lorebook manager). Unlike
// passthrough tools these are "silent bookkeeping" the story never
// references — but the user still wants transparency into what the model
// consulted/recorded. Each entry is {verb, arg} where ``arg`` names the
// meta.args key to append (e.g. "Recalling · Elena"). Covers both the
// dot-named native surface and the underscore legacy surface.
const _NARRATIVE_TOOL_LABELS = {
  recall_entity:      { verb: 'Recalling',        arg: 'name'  },
  recall_facts:       { verb: 'Consulting facts', arg: 'about' },
  recall_plot_thread: { verb: 'Checking the plot', arg: 'name' },
  recall_archive:     { verb: 'Searching past scenes', arg: 'query' },
  list_entities:      { verb: 'Reviewing the cast', arg: null  },
  'lorebook.check':   { verb: 'Checking lore',    arg: 'query' },
  lorebook_check:     { verb: 'Checking lore',    arg: 'query' },
  lorebook_search:    { verb: 'Searching lore',   arg: 'query' },
  'lorebook.create':  { verb: 'Recording lore',   arg: 'name'  },
  lorebook_create:    { verb: 'Recording lore',   arg: 'name'  },
  'lorebook.update':  { verb: 'Revising lore',    arg: 'name'  },
  lorebook_update:    { verb: 'Revising lore',    arg: 'name'  },
  'lorebook.delete':  { verb: 'Removing lore',    arg: 'name'  },
  lorebook_delete:    { verb: 'Removing lore',    arg: 'name'  },
};

/**
 * Build a short human label for one narrative tool activity entry.
 * ``args`` is the parsed meta.args dict (may be absent). Returns a plain
 * string; caller escapes before inserting into HTML.
 */
function _narrativeActivityLabel(tool, args) {
  const cfg = _NARRATIVE_TOOL_LABELS[tool] || { verb: tool, arg: null };
  let detail = '';
  if (cfg.arg && args && typeof args === 'object') {
    const v = args[cfg.arg];
    if (v != null && String(v).trim()) detail = String(v).trim();
  }
  return detail ? `${cfg.verb} · ${detail}` : cfg.verb;
}

// -----------------------------------------------------------------------
// Unified tool card visual language
// -----------------------------------------------------------------------
// Color per tool category. Search → accent-blue, fetch/read → teal,
// compute → amber, create/build → lavender, misc → slate.
const _TOOL_COLOR_BY_CATEGORY = {
  search: '#6c8aff',
  fetch:  '#06b6d4',
  compute: '#f59e0b',
  create: '#a78bfa',
  misc:   '#94a3b8',
};
const _TOOL_CATEGORY = {
  web_search: 'search', web: 'search', youtube_search: 'search',
  image_search: 'search', file_search: 'search', search_files: 'search',
  web_fetch: 'fetch', read_file: 'fetch', document_parse: 'fetch',
  youtube: 'fetch', youtube_transcript: 'fetch', wikipedia: 'fetch',
  calculator: 'compute', python_exec: 'compute', python_executor: 'compute',
  math_verify: 'compute', unit_converter: 'compute', datetime: 'compute',
  json_tool: 'compute', hash_tool: 'compute', hash: 'compute',
  app_builder: 'create', image_generation: 'create',
  create_document: 'create', create_presentation: 'create',
  create_spreadsheet: 'create', create_chart: 'create', create_ebook: 'create',
  draft_section: 'create',
};
function _toolColor(name) {
  const cat = _TOOL_CATEGORY[name] || 'misc';
  return _TOOL_COLOR_BY_CATEGORY[cat];
}

const _TOOL_CATEGORY_LABELS = {
  search: 'Search',
  fetch: 'Read',
  compute: 'Compute',
  create: 'Create',
  misc: 'Tool',
};
function _toolCategoryLabel(name) {
  return _TOOL_CATEGORY_LABELS[_TOOL_CATEGORY[name] || 'misc'];
}

// Inline stroke icons sized 14×14 — category-specific so scanning feels
// intentional. Keep stroke-width consistent at 1.6 to match app rhythm.
const _TOOL_ICONS_SVG = {
  search: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="9" cy="9" r="5.5"/><path d="m14 14 3.5 3.5"/></svg>',
  fetch:  '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 6h12M4 10h12M4 14h7"/><path d="m15 14 2 2-2 2"/></svg>',
  compute: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="4" y="4" width="12" height="12" rx="2"/><path d="M7.5 8.5h2M10.5 8.5h2M7.5 12.5h5"/></svg>',
  create: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 3v14M3 10h14"/></svg>',
  misc:   '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="10" cy="10" r="7"/><path d="M10 6v4l2.5 2.5"/></svg>',
  check:  '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m5 10.5 3.5 3.5L15.5 7"/></svg>',
  error:  '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 6l8 8M14 6l-8 8"/></svg>',
};
function _toolIcon(name) {
  return _TOOL_ICONS_SVG[_TOOL_CATEGORY[name] || 'misc'];
}

function _formatDuration(ms) {
  if (!ms || ms < 0) return '';
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

/** Compact token formatter — 12345 → "12.3K", 1500000 → "1.5M".
 * Small enough to skim alongside speed numbers; full count remains
 * in the tooltip for anyone who needs the exact value. */
function _fmtTokenCount(n) {
  if (!n || n < 0) return '0';
  if (n < 1000) return String(n);
  if (n < 1_000_000) {
    const v = n / 1000;
    return v >= 10 ? `${Math.round(v)}K` : `${v.toFixed(1)}K`;
  }
  const v = n / 1_000_000;
  return v >= 10 ? `${Math.round(v)}M` : `${v.toFixed(1)}M`;
}

function _buildGenerationStatParts(stats = {}) {
  const parts = [];
  if (stats.ttftMs > 0) {
    // Compact "250ms ttft" instead of "TTFT 250ms" — leading number
    // skims faster when scanning many messages.
    parts.push(`${_formatDuration(stats.ttftMs)} ttft`);
  }
  if (stats.tps > 0) {
    parts.push(`${stats.tps} tok/s`);
  }
  // Drop the "promptTokens + evalTokens" display. llama-server's
  // prompt_tokens is the *freshly-evaluated* count (delta), not the
  // cumulative prompt size — once KV-cache reuse kicks in on turn 2+
  // it collapses to ~4 and reads as if the model lost the conversation.
  // The cumulative size is already conveyed by the "% ctx" chip below
  // (now correct after _context_usage_payload was fixed to sum
  // prompt_n + cache_n). The delta and the per-piece breakdown live in
  // the hover title for anyone debugging cache behavior.
  if (stats.evalTokens > 0) {
    const eMark = stats.evalTokensEstimated ? '~' : '';
    parts.push(`${eMark}${_fmtTokenCount(stats.evalTokens)} tok`);
  }
  // Reasoning/CoT token chip — only shown when the model actually
  // emitted reasoning tokens (DeepSeek V4 thinking, GPT-5.x reasoning,
  // o1/o3). Sibling to evalTokens; clarifies how much of the bill
  // went to hidden chain-of-thought vs visible output.
  if (stats.reasoningTokens > 0) {
    parts.push(`${_fmtTokenCount(stats.reasoningTokens)} reasoning`);
  }
  if (stats.contextLen > 0 && stats.contextUsed > 0) {
    const pct = Math.round((stats.contextUsed / stats.contextLen) * 100);
    parts.push(`${pct}% ctx`);
  }
  // Cache-hit ratio chip. Only emit when both halves were reported
  // (provider actually splits hits vs misses). For DeepSeek that means
  // every turn after the first; for llama-server it means cache reuse
  // kicked in on a multi-turn slot.
  // Gate on the DENOMINATOR, not the hit count. Requiring cached > 0 meant
  // a 0% cache rendered nothing at all — identical to a provider with no
  // cache telemetry — so a provider silently re-charging the full prompt
  // every turn was invisible. Whenever either half was reported we know the
  // provider is cache-aware, so 0% is a real, reportable result.
  if ((stats.promptTokensEvaluated + stats.promptTokensCached) > 0) {
    const total = stats.promptTokensEvaluated + stats.promptTokensCached;
    const pct = Math.round((stats.promptTokensCached / total) * 100);
    parts.push(`${pct}% cached`);
  }
  return parts;
}

/** Render the knowledge-pack chip shown in an assistant message footer.
 *
 * Four visible outcomes:
 *   - "injected"     → "📚 Searched packs — N of M sources"   (positive grounding signal)
 *   - "all_dropped"  → "📚 Searched packs — found M, none fit context budget"  (loud failure)
 *   - "no_results"   → "📚 Searched packs — no matches"        (retrieval ran, found nothing)
 *   - "search_failed"→ "📚 Pack search failed"                 (rare, infrastructure)
 *
 * "no_bindings" / "mode_disabled" / etc. don't reach the UI — they're a true
 * no-op and produce no chip. Only attempted-retrieval results render.
 *
 * When ``top_sources`` carries a browseable URL (currently ZIM-backed packs
 * only), the chip becomes a button — click opens the source article in
 * Browse via openInBrowse. Plain references (augpack chunks without a
 * standalone view) remain text-only.
 */
function _renderKnowledgePackChip(pack) {
  if (!pack || typeof pack !== 'object') return '';
  const outcome = pack.outcome || '';
  const packs = Array.isArray(pack.packs_searched) ? pack.packs_searched : [];
  const packLabel = packs.length === 1
    ? _shortPackLabel(packs[0])
    : `${packs.length} packs`;
  const found = pack.results_found || 0;
  const used = pack.results_injected || 0;
  const dropped = pack.results_dropped_oversized || 0;
  const budget = pack.budget_chars || 0;
  const sources = Array.isArray(pack.top_sources) ? pack.top_sources : [];
  const browseable = sources.find(s => s && s.is_browseable && s.url);

  let label = '';
  let cls = 'message-pack-chip';
  let title = '';
  if (outcome === 'injected') {
    label = `Searched ${packLabel} — ${used} of ${found} sources`;
    title = `Pack retrieval grounded this response. ${used} chunk${used === 1 ? '' : 's'} fit the ${budget.toLocaleString()}-char budget; ${dropped} dropped as oversized.${browseable ? ' Click to open the top source in Browse.' : ''}`;
  } else if (outcome === 'all_dropped') {
    label = `Searched ${packLabel} — ${found} matches, none fit context`;
    cls += ' message-pack-chip--warn';
    title = `Retrieval found ${found} matches but every result exceeded the ${budget.toLocaleString()}-char budget. The model answered without pack content.${browseable ? ' Click to open the top match in Browse — the article is there even though it didn\'t fit the chat budget.' : ' Try a more specific question, or raise the per-mode result budget.'}`;
  } else if (outcome === 'no_results') {
    label = `Searched ${packLabel} — no matches`;
    title = `Retrieval ran against ${packs.length || 1} pack${packs.length === 1 ? '' : 's'} but found no relevant content for this query.`;
  } else if (outcome === 'search_failed') {
    label = `Pack search failed`;
    cls += ' message-pack-chip--warn';
    title = `Retrieval threw an exception. Check server logs for knowledge_pack_search_failed.`;
  } else {
    return '';
  }

  // Clickable when a browseable source is available; otherwise plain div.
  // Using a button for the clickable path so keyboard users get focus +
  // Enter activation for free, matching the rest of the message footer.
  if (browseable) {
    cls += ' message-pack-chip--clickable';
    return `<button type="button" class="${cls}" title="${escapeHtml(title)}" data-pack-source="${escapeHtml(browseable.url)}">📚 ${escapeHtml(label)}</button>`;
  }
  return `<div class="${cls}" title="${escapeHtml(title)}">📚 ${escapeHtml(label)}</div>`;
}

/** Compact pack id for chip display ("mdwiki_en_all_2025-11" → "mdwiki en all"). */
function _shortPackLabel(packId) {
  if (!packId) return 'pack';
  const stripped = packId.replace(/_\d{4}-\d{2}$/, '').replace(/_/g, ' ');
  return stripped.length > 24 ? stripped.slice(0, 22) + '…' : stripped;
}

function _generationStatsTitle(stats = {}) {
  const parts = ['Generation statistics'];
  if (stats.ttftMs > 0) {
    parts.push(`TTFT ${_formatDuration(stats.ttftMs)}`);
  }
  if (stats.totalDurationMs > 0) {
    parts.push(`total ${_formatDuration(stats.totalDurationMs)}`);
  }
  if (stats.evalDurationMs > 0) {
    parts.push(`generation ${_formatDuration(stats.evalDurationMs)}`);
  }
  // Tooltip shows the full picture, including the cache hit/miss
  // breakdown that the inline chip intentionally hides to stay scannable.
  // promptTokensEvaluated = freshly evaluated this turn (delta).
  // promptTokensCached    = served from the slot's KV cache (free).
  // contextUsed           = sum of the two = cumulative prompt size.
  if (stats.contextUsed > 0 || stats.evalTokens > 0) {
    const cu = (stats.contextUsed || stats.promptTokens || 0).toLocaleString();
    const e = (stats.evalTokens || 0).toLocaleString();
    parts.push(`${cu} prompt + ${e} gen tokens`);
  }
  if (stats.promptTokensEvaluated > 0 || stats.promptTokensCached > 0) {
    const fresh = (stats.promptTokensEvaluated || 0).toLocaleString();
    const cached = (stats.promptTokensCached || 0).toLocaleString();
    let line = `prompt: ${fresh} fresh + ${cached} from KV cache`;
    // Cache WRITES bill above the fresh rate (Anthropic ~1.25x), so a turn
    // that writes a lot and reads nothing is more expensive than no caching
    // at all. Surface it or that failure mode stays invisible.
    if (stats.promptTokensCacheWrite > 0) {
      line += ` + ${stats.promptTokensCacheWrite.toLocaleString()} written to cache`;
    }
    parts.push(line);
  }
  // KV reuse-audit verdict — only surfaced when reuse was actually
  // forfeited (payload divergence, server-side void, partial reuse).
  // "hot" and expected-cold turns stay silent to keep the tooltip lean.
  if (stats.kvReuse && stats.kvReuse !== 'hot' && stats.kvReuse !== 'cold_expected') {
    const cause = stats.kvVoidCause ? ` (${stats.kvVoidCause})` : '';
    parts.push(`KV reuse lost: ${stats.kvReuse.replace(/_/g, ' ')}${cause}`);
  }
  // Reasoning tokens are a subset of generation cost — surface
  // separately so users can see "of the N output tokens, M were
  // hidden CoT". Hot signal on DeepSeek V4 (Pro thinking is 4× CoT
  // volume of legacy reasoner) and GPT-5.x reasoning models.
  if (stats.reasoningTokens > 0) {
    parts.push(`${stats.reasoningTokens.toLocaleString()} reasoning tokens`);
  }
  if (stats.contextLen > 0 && stats.contextUsed > 0) {
    parts.push(`ctx ${stats.contextUsed.toLocaleString()}/${stats.contextLen.toLocaleString()}`);
  }
  if (stats.promptTokensEstimated || stats.evalTokensEstimated) {
    parts.push('~ marks tokenizer estimate (upstream omitted usage)');
  }
  return parts.join(' \u00B7 ');
}

/** Friendly names for passthrough tool indicators. */
const _PT_TOOL_LABELS = {
  web: 'Web Search', wikipedia: 'Wikipedia', youtube_transcript: 'YouTube',
  calculator: 'Calculator', datetime: 'Date & Time', unit_converter: 'Units',
  text_analysis: 'Text Analysis', json_tool: 'JSON', hash: 'Hash',
  python_exec: 'Python', file_ops: 'Files', document_parse: 'Document',
  image_generation: 'Image Gen', image_search: 'Image Search',
  draft_section: 'Drafting', create_document: 'Document',
  create_presentation: 'Slides', create_spreadsheet: 'Spreadsheet',
  create_chart: 'Chart', export_markdown: 'Markdown',
  export_csv: 'CSV', export_code: 'Code File',
};

// Wave-friendly running verbs for agentic chain steps (a "wave" is usually a
// parallel batch). Falls back to the singular tool-status label, then a
// generic "Using <noun>". Used by chainToolRunning/chainToolDone.
const _CHAIN_VERBS = {
  web_search: 'Researching the web', web_fetch: 'Reading sources',
  image_search: 'Finding images', image_generation: 'Generating images',
  chart: 'Building charts', create_chart: 'Building charts',
  python_exec: 'Running code', python_executor: 'Running code',
};
function _chainVerb(tool) {
  return _CHAIN_VERBS[tool] || _TOOL_STATUS_LABELS[tool]
    || ('Using ' + (_PT_TOOL_LABELS[tool] || tool.replace(/_/g, ' ')));
}
function _chainNoun(tool) {
  return _PT_TOOL_LABELS[tool] || tool.replace(/_/g, ' ');
}

// ---------------------------------------------------------------------------
// Avatar helpers — minimal. Chat is the tool surface; persona/presence work
// is owned by the companion (Becca) which has its own surfaces with lipsync
// TTS. So we only render an avatar when there's a REAL image to render:
//   - User w/ `session.userAvatar` URL  → that image
//   - Assistant in NARRATIVE mode w/ `session.characterAvatar` URL → that
//     image (Sillytavern-style persona; the assistant genuinely is a
//     different character per chat here)
//   - Everything else → no avatar at all (`message--no-avatar` class
//     collapses the slot in CSS)
//
// No glyphs, no initials, no fallback gradient. This intentionally tracks
// Claude.ai / ChatGPT — chrome out of the way of the content. The reactive
// gradient bubble + mode glyphs from the prior pass are gone.
// ---------------------------------------------------------------------------

/** Build avatar HTML for a message bubble. Returns the markup that goes
 *  inside `<div class="message-avatar">…</div>`. Empty string means "no
 *  avatar this message" — the caller should add `message--no-avatar` to
 *  the parent `.message` so CSS collapses the slot. */
function _buildAvatarInnerHtml(role, session) {
  const isUser = role === 'user';
  const isNarrative = session?.mode === 'narrative';
  const realSrc = isUser
    ? (session?.userAvatar || '')
    : (isNarrative ? (session?.characterAvatar || '') : '');
  if (!realSrc) return '';
  const alt = isUser ? 'You' : (session?.title || 'Assistant');
  return `<img src="${escapeHtml(realSrc)}" alt="${escapeHtml(alt)}" onerror="this.remove()">`;
}

// ---------------------------------------------------------------------------
// Streaming-render helpers
// ---------------------------------------------------------------------------

function _prefersReducedMotion() {
  return typeof window !== 'undefined'
      && typeof window.matchMedia === 'function'
      && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}


// ---------------------------------------------------------------------------
// MessageRenderer
// ---------------------------------------------------------------------------

export class MessageRenderer {

  _renderStoredToolCards(messageEl, cards) {
    if (!messageEl || !Array.isArray(cards) || !cards.length) return;
    const content = messageEl.querySelector('.message-content');
    const host = this._ensureToolCardHost(content);
    if (!host) return;
    import('./tool-card.js').then(m => {
      cards.forEach((card) => {
        const artifactId = String(card?.artifact_id || '');
        if (artifactId && host.querySelector(`.tool-card--typed[data-artifact-id="${CSS.escape(artifactId)}"]`)) {
          return;
        }
        const html = m.renderToolCard(card);
        if (!html) return;
        const wrap = document.createElement('div');
        wrap.innerHTML = html;
        if (wrap.firstElementChild) host.appendChild(wrap.firstElementChild);
      });
    }).catch(() => { /* leave message body intact */ });
  }

  /**
   * @param {object} options
   * @param {function} options.onAction  - (action, nodeId, data) => void
   * @param {string}   options.mode      - Current UI mode (passthrough, narrative, etc.)
   * @param {object}   options.highlightHooks - Post-processing hooks for highlightCode
   */
  constructor(options = {}) {
    // DOM references — created by createDOM()
    this.messagesEl   = null;   // .chat-messages container
    this.scrollEl     = null;   // .chat-scroll wrapper
    this.emptyStateEl = null;   // .empty-state element
    this.streamingEl  = null;   // Current streaming message (replaces global #streaming-message)

    // Per-instance scroll tracking
    this.userScrolledUp = false;

    // Stream state — per-instance, NOT global
    this._streamPhases       = [];
    this._streamToolCalls    = [];
    // Unified-event tool cards (tool_start/progress/complete). Captured
    // here so finalizeStreaming can persist them on the assistant node;
    // without persistence the cards vanish on the next renderMessages()
    // pass (next turn, branch nav, or page refresh).
    this._streamUnifiedToolCards = [];
    // Narrative internal-tool activity (recall + lorebook) for the live
    // trail on the streaming message. Snapshotted onto the assistant node
    // by finalizeStreaming (collectNarrativeActivity) so the collapsed chip
    // survives renderMessages() on that message; full history lives in the
    // inspector request-log, not the main transcript.
    this._streamNarrativeActivity = [];
    this._streamThinking     = '';
    this._streamThinkingOpen = false;
    this._streamPhaseContent = {};
    this._streamComplexity   = '';
    this._streamFlowName     = '';
    this._streamSpeaker      = '';  // Group-chat speaker for the in-flight turn
    this._streamMetrics      = {
      tps: 0,
      contextLen: 0,
      contextUsed: 0,
      promptTokens: 0,
      promptTokensEvaluated: 0,
      promptTokensCached: 0,
      promptTokensCacheWrite: 0,
      kvReuse: '',
      kvVoidCause: '',
      evalTokens: 0,
      reasoningTokens: 0,
      ttftMs: 0,
      totalDurationMs: 0,
      evalDurationMs: 0,
    };

    // Streaming render coalescer (rAF) + stable/active split bookkeeping.
    // Without these, every NDJSON delta would re-render the full message
    // bubble and re-run hljs over every existing code block.
    this._streamRenderScheduled = false;
    this._streamSplit = newSplitState();  // stable/active + open-fence bookkeeping
    // Parallel coalescer for the reasoning/thinking stream. Reasoning models
    // (DeepSeek, Qwen3, GLM, GPT-OSS) emit thinking down a SEPARATE path from
    // content; without this it re-wrote the full accumulated thinking string +
    // smooth-scrolled on every token — O(n²) + a compositor animation per
    // delta, the freeze on long CoT. Mirrors _streamRenderScheduled.
    this._streamThinkRenderScheduled = false;

    // Callbacks
    this._onAction       = options.onAction || (() => {});
    this._mode           = options.mode || 'passthrough';
    this._highlightHooks = options.highlightHooks || {};

    // Narrative panel default-collapsed state. Driven by the active
    // character's autoCollapseNarrativePanels preference; default true.
    // Set via setNarrativePanelsCollapsed() when the active session/char
    // changes.
    this._narrativePanelsCollapsed = true;

    // Bound handler references (for cleanup)
    this._boundScrollHandler = null;
    this._boundClickHandler  = null;
  }

  // -----------------------------------------------------------------------
  // DOM creation
  // -----------------------------------------------------------------------

  /**
   * Build the chat DOM structure and append it to `container`.
   * @param {HTMLElement} container
   */
  createDOM(container) {
    // Scroll wrapper
    this.scrollEl = document.createElement('div');
    this.scrollEl.className = 'chat-scroll';

    // Messages container
    this.messagesEl = document.createElement('div');
    this.messagesEl.className = 'chat-messages';

    this.scrollEl.appendChild(this.messagesEl);
    container.appendChild(this.scrollEl);

    // Empty state — mode-aware
    this.emptyStateEl = document.createElement('div');
    this.emptyStateEl.className = 'empty-state';
    this.emptyStateEl.innerHTML = this._buildEmptyState();
    this.messagesEl.appendChild(this.emptyStateEl);

    this._initScrollTracking();
    this._initDelegatedActions();
  }

  // -----------------------------------------------------------------------
  // Empty state (mode-aware)
  // -----------------------------------------------------------------------

  _buildEmptyState() {
    const configs = {
      analytical: {
        icon: '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
        title: 'What would you like to analyze?',
        color: 'var(--orb-analytical, #2196f3)',
        chips: [
          { prompt: 'Analyze the pros and cons of this approach', label: 'Compare options' },
          { prompt: 'Break down this problem step by step', label: 'Step-by-step analysis' },
          { prompt: 'Research and summarize the latest findings on this topic', label: 'Deep research' },
          { prompt: 'Review this data and identify key patterns', label: 'Pattern analysis' },
        ],
      },
      agentic: {
        icon: '<rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>',
        title: 'What would you like to build?',
        color: 'var(--orb-agentic, #ff9800)',
        chips: [
          { prompt: 'Build a landing page with a modern design', label: 'Landing page' },
          { prompt: 'Create a simple web app with a form and database', label: 'Web app' },
          { prompt: 'Build an interactive dashboard with charts', label: 'Dashboard' },
          { prompt: 'Create a tool that automates a repetitive task', label: 'Automation tool' },
        ],
      },
      narrative: {
        icon: '<path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5z"/>',
        title: 'Begin your story',
        color: 'var(--orb-narrative, #9c27b0)',
        chips: [
          { prompt: 'Start a fantasy adventure in a magical kingdom', label: 'Fantasy adventure' },
          { prompt: 'Begin a mystery set in a noir detective world', label: 'Mystery noir' },
          { prompt: 'Create a sci-fi story on a generation ship', label: 'Sci-fi voyage' },
          { prompt: 'Start a slice-of-life story in a small town', label: 'Slice of life' },
        ],
      },
      passthrough: {
        icon: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
        title: 'What would you like to do?',
        color: 'var(--orb-passthrough, #4caf50)',
        chips: [
          { prompt: 'Search the web for today\'s top news', label: 'Search the web' },
          { prompt: 'Explain how transformers work in machine learning', label: 'Explain a concept' },
          { prompt: 'Write a Python script that reads a CSV and generates a summary report', label: 'Write some code' },
          { prompt: 'Help me brainstorm ideas for a weekend project', label: 'Brainstorm ideas' },
        ],
      },
    };
    const c = configs[this._mode] || configs.passthrough;
    const chipsHtml = c.chips.map(ch =>
      `<button class="empty-state-chip" data-prompt="${ch.prompt.replace(/"/g, '&quot;')}">${ch.label}</button>`
    ).join('');

    return `
      <div class="empty-state-icon" style="color: ${c.color}">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          ${c.icon}
        </svg>
      </div>
      <p class="empty-state-title">${c.title}</p>
      <div class="empty-state-chips">${chipsHtml}</div>
    `;
  }

  // -----------------------------------------------------------------------
  // Scroll tracking (per-instance)
  // -----------------------------------------------------------------------

  _initScrollTracking() {
    this._boundScrollHandler = () => {
      const el = this.scrollEl;
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      this.userScrolledUp = !atBottom;
    };
    this.scrollEl.addEventListener('scroll', this._boundScrollHandler);

    // iOS keyboard handler. When the soft keyboard opens, the visual
    // viewport shrinks but the layout viewport doesn't, so the last
    // message the user was replying to can disappear behind the
    // keyboard. Re-snap to the bottom on any visualViewport resize,
    // but only when the user wasn't already scrolled up (so we don't
    // yank them away from reading older messages mid-scroll).
    //
    // visualViewport is missing on some older browsers; guard so the
    // rest of the renderer still works without it.
    if (typeof window !== 'undefined' && window.visualViewport) {
      this._boundViewportResize = () => {
        if (!this.scrollEl || this.userScrolledUp) return;
        // rAF lets layout recompute after the viewport change —
        // scrollHeight measured before the resize would be stale.
        requestAnimationFrame(() => this.scrollToBottom(false, true));
      };
      window.visualViewport.addEventListener(
        'resize', this._boundViewportResize,
      );
    }

    // Window resize / orientation change: re-anchor to the bottom when the
    // user wasn't scrolled up. visualViewport above covers iOS keyboard;
    // this covers desktop window-resize and mobile orientation flips,
    // which re-wrap text and shift message heights — without re-anchoring
    // the user falls "above" the new bottom and sees the chat jump.
    // RAF-gate to coalesce drag-resize bursts.
    if (typeof window !== 'undefined') {
      let pending = false;
      this._boundWindowResize = () => {
        if (!this.scrollEl || pending) return;
        pending = true;
        requestAnimationFrame(() => {
          pending = false;
          if (!this.userScrolledUp) this.scrollToBottom(false, true);
        });
      };
      window.addEventListener('resize', this._boundWindowResize);
    }
  }

  // -----------------------------------------------------------------------
  // Delegated actions (per-instance, on messagesEl)
  // -----------------------------------------------------------------------

  _initDelegatedActions() {
    this._boundClickHandler = (e) => {
      // Cast-to-TV overlay on chat-attached images. Must check BEFORE
      // the lightbox handler so the overlay button doesn't also trigger
      // lightbox opening. Dynamic import keeps cast-picker out of the
      // chat module's static dep set (renderer is hot path, picker is
      // cold path).
      const castImgBtn = e.target.closest('[data-action="cast-image"]');
      if (castImgBtn) {
        e.preventDefault();
        e.stopPropagation();
        const src = castImgBtn.dataset.src || '';
        if (!src) return;
        import('../cast-picker.js').then(({ openCastPicker }) => {
          openCastPicker({
            anchor: castImgBtn,
            capability: 'display.image_show@1',
            content: {
              contentUrl: src,
              title: 'Chat image',
              contentKey: src,
              metadata: { source: 'chat-image' },
            },
          });
        }).catch((err) => console.warn('[chat-renderer] cast picker import failed', err));
        return;
      }

      // Markdown images — open lightbox via action callback
      const mdImg = e.target.closest('.md-image');
      if (mdImg) {
        this._onAction('lightbox', null, { src: mdImg.src });
        return;
      }

      // Narrative panel toggle (collapsed/expanded). Per-element only —
      // no persistence needed; CSS reads aria-expanded directly.
      const panelToggle = e.target.closest('.narrative-panel-toggle');
      if (panelToggle) {
        const expanded = panelToggle.getAttribute('aria-expanded') === 'true';
        panelToggle.setAttribute('aria-expanded', expanded ? 'false' : 'true');
        return;
      }

      // Stall banner — Abort & retry. Aborting + re-streaming the
      // partial is owned by index.js because it has the session +
      // activeSession refs; we just fire the intent.
      const stallBtn = e.target.closest('[data-action="stall-abort-retry"]');
      if (stallBtn) {
        e.stopPropagation();
        document.dispatchEvent(new CustomEvent('augmentum:stall-abort-retry'));
        return;
      }

      // Action overflow trigger — opens the more-actions popover anchored
      // to the trigger button. Sibling state (last-assistant / narrative
      // mode) is read from the DOM + renderer state so we don't need a
      // session ref here. Stops here so the click doesn't also fire the
      // popover's outside-handler.
      const overflowBtn = e.target.closest('[data-action="actions-overflow"]');
      if (overflowBtn) {
        e.stopPropagation();
        e.preventDefault();
        if (overflowBtn.getAttribute('aria-expanded') === 'true') {
          this._closeActionOverflow();
        } else {
          const nodeId = overflowBtn.dataset.nodeId;
          const messageEl = overflowBtn.closest('.message');
          // Last assistant = no .message-assistant comes after this one
          // in the messages container.
          let isLastAssistant = false;
          if (messageEl && this.messagesEl) {
            const all = this.messagesEl.querySelectorAll('.message.message-assistant');
            isLastAssistant = all.length > 0 && all[all.length - 1] === messageEl;
          }
          const isNarrativeSession = this._mode === 'narrative';
          this._openActionOverflow(overflowBtn, nodeId, {
            isLastAssistant,
            isNarrativeSession,
          });
        }
        return;
      }

      // Rerun-as: drill from the overflow into a mode picker (replaces
      // the legacy window.prompt). The picker reuses the overflow slot
      // — opening it implicitly closes the current overflow. Anchored
      // to whatever was anchoring the overflow (the `…` button).
      const rerunPickBtn = e.target.closest('[data-action="rerun-as-pick"]');
      if (rerunPickBtn) {
        e.stopPropagation();
        e.preventDefault();
        const nodeId = rerunPickBtn.dataset.nodeId;
        const messageEl = rerunPickBtn.closest('.message');
        const content = messageEl?.dataset.rawContent || '';
        // Anchor to the original `…` trigger so the picker doesn't drift
        // when the previous row is removed from DOM mid-replace.
        const anchorBtn = (this._actionOverflow?.triggerBtn) || rerunPickBtn;
        this._openModePickerPopover(anchorBtn, nodeId, this._mode, content);
        return;
      }

      // Rerun-as: mode picked. Dispatch with targetMode and close.
      const rerunGoBtn = e.target.closest('[data-action="rerun-as-go"]');
      if (rerunGoBtn) {
        e.stopPropagation();
        e.preventDefault();
        const nodeId = rerunGoBtn.dataset.nodeId;
        const targetMode = rerunGoBtn.dataset.targetMode;
        const popEl = rerunGoBtn.closest('.action-overflow-popover');
        const content = popEl?.dataset.rerunContent || '';
        this._closeActionOverflow();
        this._onAction('rerun-as', nodeId, { content, button: rerunGoBtn, targetMode });
        return;
      }

      // Any other click inside the overflow popover that lands on an
      // actionable row — close the popover so the next render rebuilds
      // cleanly. The actual action handler is one of the cases below
      // (copy/tts/breakdown/edit/etc.) — they all use class or
      // data-action selectors that work regardless of DOM location.
      if (e.target.closest('.action-overflow-popover .action-overflow-row')) {
        // Don't return — let the existing handler fire AND close the popover.
        this._closeActionOverflow();
      }

      // Inspector button — stored-reasoning footer. Bound BEFORE the
      // generic header toggle so the inspector click doesn't also
      // collapse the panel. window.__inspectStoredReasoning is wired in
      // chat/index.js.
      const inspectBtn = e.target.closest('.reasoning-summary__inspect');
      if (inspectBtn) {
        e.stopPropagation();
        if (typeof window.__inspectStoredReasoning === 'function') {
          window.__inspectStoredReasoning(inspectBtn);
        }
        return;
      }

      // Thinking-panel / reasoning-summary headers — toggle the parent's
      // .open class. Replaces the old inline onclick handlers; gets
      // keyboard activation for free because the markup uses <button>.
      // ARIA expanded state stays in sync with the class.
      const thinkingToggle = e.target.closest('.thinking-header, .reasoning-summary__header');
      if (thinkingToggle) {
        const parent = thinkingToggle.parentElement;
        if (parent) {
          const willOpen = !parent.classList.contains('open');
          parent.classList.toggle('open');
          thinkingToggle.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
        }
        return;
      }

      // Knowledge-pack chip — open the source article in Browse. Action
      // dispatch goes through the renderer's onAction so the host
      // (chat/index.js) decides whether to import browse.js dynamically
      // (avoids a hard dep from chat onto browse for users who never
      // install a pack).
      const packSource = e.target.closest('[data-pack-source]');
      if (packSource) {
        const url = packSource.dataset.packSource;
        if (url) this._onAction('open-in-browse', null, { url });
        return;
      }

      // Branch navigation
      const branchPrev = e.target.closest('[data-action="branch-prev"]');
      if (branchPrev) {
        this._onAction('branch', branchPrev.dataset.nodeId, -1);
        return;
      }
      const branchNext = e.target.closest('[data-action="branch-next"]');
      if (branchNext) {
        this._onAction('branch', branchNext.dataset.nodeId, +1);
        return;
      }

      // Regenerate
      const regenAction = e.target.closest('[data-action="regenerate-message"]');
      if (regenAction) {
        const nid = regenAction.dataset.nodeId;
        if (nid) this._onAction('regenerate', nid);
        return;
      }

      // Continue assistant message
      const continueAction = e.target.closest('[data-action="continue-message"]');
      if (continueAction) {
        const nid = continueAction.dataset.nodeId;
        if (nid) this._onAction('continue', nid);
        return;
      }

      // Impersonate (AI writes as user)
      const impersonateAction = e.target.closest('[data-action="impersonate-message"]');
      if (impersonateAction) {
        const nid = impersonateAction.dataset.nodeId;
        this._onAction('impersonate', nid);
        return;
      }

      // Edit message
      const editAction = e.target.closest('[data-action="edit-message"]');
      if (editAction) {
        this._onAction('edit', editAction.dataset.nodeId);
        return;
      }

      // Save edit
      const saveEdit = e.target.closest('[data-action="save-edit"]');
      if (saveEdit) {
        const nid = saveEdit.dataset.nodeId;
        const msgEl = saveEdit.closest('.message');
        const textarea = msgEl?.querySelector('.edit-textarea');
        if (nid && textarea) {
          const newContent = textarea.value.trim();
          if (newContent) this._onAction('save-edit', nid, { content: newContent });
        }
        return;
      }

      // Cancel edit
      const cancelEdit = e.target.closest('[data-action="cancel-edit"]');
      if (cancelEdit) {
        this._onAction('cancel-edit', null);
        return;
      }

      // Delete message
      const deleteAction = e.target.closest('[data-action="delete-message"]');
      if (deleteAction) {
        this._onAction('delete', deleteAction.dataset.nodeId);
        return;
      }

      // Share with Becca (narrative → companion memory graduation; Lane 3 §4.6).
      // ONLY content-crossing path from narrative into Becca's memory — the
      // companion will labeler-process the graduated content as if witnessed.
      // Same body-appended popover caveat as TTS/breakdown above — fall back
      // to a nodeId-keyed document lookup so the row's content reaches the
      // graduation handler instead of arriving as an empty string.
      const graduateAction = e.target.closest('[data-action="graduate-to-becca"]');
      if (graduateAction) {
        const nodeId = graduateAction.dataset.nodeId;
        const msgEl = graduateAction.closest('.message')
          || (nodeId ? document.querySelector(`.message[data-node-id="${nodeId}"]`) : null);
        const content = msgEl?.dataset.rawContent || '';
        this._onAction('graduate-to-becca', nodeId, { content, button: graduateAction });
        return;
      }

      // Rerun this turn against a different mode (Lane 3 §9 DPO pair).
      // The legacy direct `rerun-as` row was replaced by `rerun-as-pick`
      // → mode picker popover → `rerun-as-go`. The picker handler is
      // earlier in this method (it gates on `_actionOverflow` state).

      // A/B test voting
      const voteBtn = e.target.closest('.ab-vote-btn');
      if (voteBtn) {
        const voteEl = voteBtn.closest('.ab-test-vote');
        if (!voteEl) return;
        const vote       = voteBtn.dataset.vote;
        const balancerId = voteEl.dataset.balancerId;
        const model      = voteEl.dataset.model;
        const backend    = voteEl.dataset.backend;
        const nodeId     = voteEl.dataset.nodeId;

        // Visual feedback immediately
        voteEl.querySelectorAll('.ab-vote-btn').forEach(b => b.classList.remove('voted'));
        voteBtn.classList.add('voted');

        this._onAction('vote', nodeId, { vote, balancerId, model, backend });
        return;
      }

      // Message copy button (delegated — no per-element listeners)
      const copyMsgBtn = e.target.closest('.copy-msg-btn');
      if (copyMsgBtn) {
        const msgEl = copyMsgBtn.closest('.message');
        const content = msgEl?.dataset.rawContent || '';
        copyToClipboard(content).then((ok) => {
          if (!ok) return;
          copyMsgBtn.innerHTML = icons.check;
          setTimeout(() => { copyMsgBtn.innerHTML = icons.copy; }, 1500);
        });
        return;
      }

      // Message TTS button (delegated). The button can live in the
      // primary message-actions row OR in the body-appended overflow
      // popover, so closest('.message') won't always reach the source
      // message — we fall back to a nodeId-keyed document lookup.
      // Without this fix the popover variant silently no-ops because
      // ttsPlayMessage's `if (!text)` guard bails on the empty string.
      const ttsMsgBtn = e.target.closest('.tts-msg-btn');
      if (ttsMsgBtn) {
        const nodeId = ttsMsgBtn.dataset.nodeId
          || ttsMsgBtn.closest('.message')?.dataset.nodeId;
        const msgEl = ttsMsgBtn.closest('.message')
          || (nodeId ? document.querySelector(`.message[data-node-id="${nodeId}"]`) : null);
        const content = msgEl?.dataset.rawContent || '';
        this._onAction('tts', nodeId, { text: content, button: ttsMsgBtn });
        return;
      }

      // Breakdown button — opens the sentence-breakdown popover when
      // the active character is a language partner (has lang_code on
      // its data blob). Click outside a partner session flashes a
      // tooltip so the button doesn't appear inert. Lazy-imports the
      // popover so non-partner sessions never pull the bundle.
      // Same body-appended popover caveat as the TTS button above —
      // fall back to a nodeId-keyed document lookup when closest('.message')
      // can't reach the source row.
      const bdBtn = e.target.closest('.breakdown-msg-btn');
      if (bdBtn) {
        const nodeId = bdBtn.dataset.nodeId || bdBtn.closest('.message')?.dataset.nodeId;
        const msgEl = bdBtn.closest('.message')
          || (nodeId ? document.querySelector(`.message[data-node-id="${nodeId}"]`) : null);
        const content = msgEl?.dataset.rawContent || '';
        const partnerLang = window.narrative?.activeCharacter?.ttsVoiceLang || '';
        // Breakdown is now context-gated at render time (only shown when a
        // language partner is configured). The defensive bail stays as a
        // last-resort: if a stale DOM survives a partner switch, just no-op
        // silently rather than open an empty popover.
        if (!partnerLang) return;
        import('../learning_games/breakdown_popover.js')
          .then(m => m.openBreakdownPopover({
            text: content,
            lang: partnerLang,
            anchor: bdBtn,
          }))
          .catch(err => console.warn('[breakdown] open failed', err));
        return;
      }

      // Code block copy
      const copyEl = e.target.closest('[data-copy]');
      if (copyEl) {
        const text = decodeURIComponent(copyEl.dataset.copy);
        copyToClipboard(text).then((ok) => {
          if (!ok) return;
          copyEl.textContent = 'Copied!';
          setTimeout(() => { copyEl.textContent = 'Copy'; }, 1500);
        });
        return;
      }

      // Code block actions — delegate to onAction('code', ...)
      const codeActionBtn = e.target.closest('.code-action-btn[data-action]');
      if (codeActionBtn) {
        const action     = codeActionBtn.dataset.action;
        const codeHeader = codeActionBtn.closest('.code-header');
        if (action && codeHeader) {
          this._onAction('code', null, { action, codeHeader });
        }
        return;
      }

      // Empty-state chip clicks
      const chip = e.target.closest('.empty-state-chip');
      if (chip && chip.dataset.prompt) {
        this._onAction('send', null, { text: chip.dataset.prompt });
        return;
      }
    };
    this.messagesEl.addEventListener('click', this._boundClickHandler);
  }

  // -----------------------------------------------------------------------
  // Inline message editing (shared by every surface)
  // -----------------------------------------------------------------------

  /**
   * Open the inline edit textarea for a message row. Self-contained: builds
   * the editor inside THIS renderer's own messagesEl and routes Save/Cancel
   * back through the delegated `save-edit`/`cancel-edit` actions, so it works
   * identically in the primary singleton surface and in every independent
   * ChatSurface. (Previously this lived only in chat/index.js keyed to the
   * primary renderer, so edit was a dead no-op in any secondary mode.)
   * @returns {boolean} true if the editor opened.
   */
  beginInlineEdit(nodeId, content) {
    const msgEl = this.messagesEl?.querySelector(`[data-node-id="${nodeId}"]`);
    if (!msgEl) return false;
    const contentEl = msgEl.querySelector('.message-content');
    if (!contentEl) return false;
    // Don't stack editors if edit is clicked twice.
    if (msgEl.querySelector('.edit-wrap')) return true;
    const actionsEl = msgEl.querySelector('.message-actions');

    if (actionsEl) actionsEl.style.display = 'none';
    const textarea = document.createElement('textarea');
    textarea.className = 'edit-textarea';
    textarea.value = content ?? '';
    textarea.rows = 1;

    const btnRow = document.createElement('div');
    btnRow.className = 'edit-actions';
    btnRow.innerHTML = `
      <button class="btn btn-sm btn-ghost" data-action="cancel-edit">Cancel</button>
      <button class="btn btn-sm btn-primary" data-action="save-edit" data-node-id="${nodeId}">Save</button>
    `;

    contentEl.style.display = 'none';
    const editWrap = document.createElement('div');
    editWrap.className = 'edit-wrap';
    editWrap.appendChild(textarea);
    editWrap.appendChild(btnRow);
    contentEl.parentElement.insertBefore(editWrap, contentEl.nextSibling);

    const autoGrow = () => {
      textarea.style.height = 'auto';
      const cap = Math.max(160, window.innerHeight * 0.6);
      textarea.style.height = Math.min(textarea.scrollHeight, cap) + 'px';
    };
    textarea.addEventListener('input', autoGrow);
    textarea.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape') {
        ev.preventDefault();
        this._onAction('cancel-edit', null);
      } else if ((ev.ctrlKey || ev.metaKey) && ev.key === 'Enter') {
        ev.preventDefault();
        const v = textarea.value.trim();
        if (v) this._onAction('save-edit', nodeId, { content: v });
      }
    });
    textarea.focus();
    const end = textarea.value.length;
    textarea.setSelectionRange(end, end);
    autoGrow();
    return true;
  }

  /** Tear down any open inline editor(s) in this renderer, restoring content. */
  cancelInlineEdit() {
    const editWraps = this.messagesEl?.querySelectorAll('.edit-wrap');
    editWraps?.forEach((w) => {
      const contentEl = w.previousElementSibling;
      if (contentEl) contentEl.style.display = '';
      const actionsEl = w.closest('.message')?.querySelector('.message-actions');
      if (actionsEl) actionsEl.style.display = '';
      w.remove();
    });
  }

  // -----------------------------------------------------------------------
  // Render full message list
  // -----------------------------------------------------------------------

  /**
   * Render all messages for a session's active branch.
   * @param {object} session
   */
  renderMessages(session) {
    // Detach empty-state before clearing so it survives innerHTML = ''
    if (this.emptyStateEl && this.emptyStateEl.parentNode === this.messagesEl) {
      this.messagesEl.removeChild(this.emptyStateEl);
    }
    // Offer chips (chat/offer-chip.js) are live, server-backed auxiliary UI
    // the offer feed appends into THIS container — they aren't session
    // messages, so the rebuild below would destroy them. Detach and re-append
    // around the wipe (same technique as empty-state; detaching a node keeps
    // its event handlers) so a re-render — next user turn, branch nav, or the
    // session-update refresh in chat/index.js — doesn't silently drop a
    // pending proposal like the image-generation "Generate?" chip mid-flight.
    const offerChips = Array.from(this.messagesEl.children).filter(
      (c) => c.classList && c.classList.contains('offer-chip'),
    );
    for (const chip of offerChips) this.messagesEl.removeChild(chip);

    // Any open overflow popover anchors to a button we're about to wipe.
    this._closeActionOverflow();

    this.messagesEl.innerHTML = '';

    if (!session || !sessionHasMessages(session)) {
      this.messagesEl.appendChild(this.emptyStateEl);
      this.updateEmptyState(false);
      for (const chip of offerChips) this.messagesEl.appendChild(chip);
      return;
    }
    this.updateEmptyState(true);

    const path = getActivePath(session);
    for (const node of path) {
      this.messagesEl.appendChild(this.createMessageEl(node, session));
    }

    // Re-append preserved offer chips at the end of the stream — their
    // natural position, after the most recent message.
    for (const chip of offerChips) this.messagesEl.appendChild(chip);

    // Restore the ambient stats bar from the most recent assistant node
    // that carries usage. Without this the bar starts empty on every
    // page reload until the next turn lands, which feels broken when
    // you scroll up to a long completed conversation.
    for (let i = path.length - 1; i >= 0; i -= 1) {
      const n = path[i];
      if (n && n.role === 'assistant' && (n.context_used || n.eval_tokens || n.ttft_ms)) {
        document.dispatchEvent(new CustomEvent('augmentum:turn-stats', {
          detail: {
            tps: n.tokens_per_second || 0,
            evalTokens: n.eval_tokens || 0,
            promptTokens: n.prompt_tokens || 0,
            promptTokensEvaluated: n.prompt_tokens_evaluated || 0,
            promptTokensCached: n.prompt_tokens_cached || 0,
            promptTokensCacheWrite: n.prompt_tokens_cache_write || 0,
            kvReuse: n.kv_reuse || '',
            kvVoidCause: n.kv_void_cause || '',
            reasoningTokens: n.reasoning_tokens || 0,
            contextLen: n.context_length || 0,
            contextUsed: n.context_used || 0,
            ttftMs: n.ttft_ms || 0,
            totalDurationMs: n.total_duration_ms || 0,
            evalDurationMs: n.eval_duration_ms || 0,
            promptTokensEstimated: !!n.prompt_tokens_estimated,
            evalTokensEstimated: !!n.eval_tokens_estimated,
          },
        }));
        break;
      }
    }

    // Defer syntax highlighting off the critical path — for a long thread
    // with dozens of code blocks, running hljs synchronously here pushes
    // time-to-first-paint well past a second. Text + layout paint first,
    // colors fill in during the next idle slice. Streaming + per-message
    // updates stay synchronous so live code still colorises chunk-by-chunk
    // as the model types it.
    highlightCodeDeferred(this.messagesEl, this._highlightHooks);

    // Anchor to bottom and STAY anchored while the content settles —
    // deferred highlight, image loads, and font-reflow all shift heights
    // over the next second or so. Single ResizeObserver replaces the
    // previous "4× scrollToBottom + per-image load listener" dance, and
    // self-disconnects after a settling window so we don't leak. Honors
    // userScrolledUp so we don't yank a reader who jumped up mid-render.
    this._anchorScrollToBottom();
  }

  /**
   * Append ONE message node to the end of the thread without wiping and
   * re-rendering the whole conversation.
   *
   * This is the hot path for sending: rendering the new user bubble must
   * NOT re-run ``renderMarkdown`` synchronously over every prior message
   * (``renderMessages`` does — it's O(all messages)). On a long thread that
   * froze the page for seconds right after hitting send, before the
   * response even started. Use this whenever the active path only GREW by a
   * tail node; ``renderMessages`` is still correct for branch / edit /
   * delete where the path actually changes.
   *
   * @returns {HTMLElement|null} the appended element
   */
  appendMessage(node, session) {
    if (!this.messagesEl || !node) return null;
    if (this.emptyStateEl && this.emptyStateEl.parentNode === this.messagesEl) {
      this.messagesEl.removeChild(this.emptyStateEl);
      this.updateEmptyState(true);
    }
    this._closeActionOverflow();
    const el = this.createMessageEl(node, session);
    this.messagesEl.appendChild(el);
    // Highlight only the new subtree, deferred to idle — never re-walks the
    // whole thread.
    highlightCodeDeferred(el, this._highlightHooks);
    this._anchorScrollToBottom();
    return el;
  }

  /** Pin the scroll viewport to the bottom for a short settling window
   *  after a full render. Replaces the older rAF chain + per-image load
   *  listeners. Cancelled if the user scrolls up. */
  _anchorScrollToBottom() {
    if (!this.messagesEl || !this.scrollEl) return;
    this.scrollToBottom(false, true);

    // Tear down any previous anchor first — back-to-back renderMessages
    // calls (rapid session swap) shouldn't stack observers.
    this._releaseScrollAnchor();

    if (typeof ResizeObserver === 'undefined') return;
    this._anchorObserver = new ResizeObserver(() => {
      if (this.userScrolledUp) return;
      this.scrollToBottom(false, true);
    });
    this._anchorObserver.observe(this.messagesEl);

    // 2s window is enough to cover deferred hljs + most image loads
    // without keeping the observer wired forever. If a slow image still
    // shifts layout after that, the user will see at most a small jump
    // they can manually re-anchor by scrolling.
    this._anchorTimeout = setTimeout(() => this._releaseScrollAnchor(), 2000);
  }

  /** @private */
  _releaseScrollAnchor() {
    if (this._anchorObserver) {
      this._anchorObserver.disconnect();
      this._anchorObserver = null;
    }
    if (this._anchorTimeout) {
      clearTimeout(this._anchorTimeout);
      this._anchorTimeout = null;
    }
  }

  // -----------------------------------------------------------------------
  // Assistant action row — shared between createMessageEl + finalizeStreaming
  // -----------------------------------------------------------------------

  /** Inner HTML for the assistant message-actions div.
   *
   *  Layout: 5 primary slots + 1 overflow trigger.
   *    Primary:  Copy · Read aloud · Edit · Regenerate · Continue (slot-reserved
   *              when not last)
   *    Overflow: Break down · Impersonate · Rerun as · Share-with-Becca · Delete
   *
   *  Read aloud lives in the primary row because it's used often — burying it
   *  behind the overflow trigger added a click for the dominant use case.
   *
   *  The overflow popover is opened via `data-action="actions-overflow"` and
   *  built lazily in `_openActionOverflow` — the row markup stays tiny and
   *  width-stable. Continue uses the existing `--invisible` slot trick so the
   *  primary row never changes width as you scroll through a thread.
   *
   *  Class names on individual buttons (`copy-msg-btn`, `tts-msg-btn`,
   *  `breakdown-msg-btn`) are preserved so the existing delegated handlers
   *  (closest('.copy-msg-btn') etc.) keep firing whether the button is in
   *  the primary row or the overflow popover. The tts/breakdown/graduate
   *  handlers fall back to a nodeId-keyed document lookup so the body-
   *  appended popover variant still finds the source `.message` row. */
  _buildAssistantActionsInnerHtml(nodeId, opts = {}) {
    const isLastAssistant = opts.isLastAssistant !== false;
    const lastOnly = isLastAssistant ? '' : ' message-action-btn--invisible';
    const lastAttrs = isLastAssistant ? '' : ' aria-hidden="true" tabindex="-1"';
    return `
        <button class="message-action-btn copy-msg-btn" title="Copy">${icons.copy}</button>
        <button class="message-action-btn tts-msg-btn" data-node-id="${nodeId}" title="Read aloud" aria-label="Read aloud">${icons.speaker}</button>
        <button class="message-action-btn" data-action="edit-message" data-node-id="${nodeId}" title="Edit">${icons.edit}</button>
        <button class="message-action-btn" data-action="regenerate-message" data-node-id="${nodeId}" title="Regenerate">${icons.regen}</button>
        <button class="message-action-btn${lastOnly}" data-action="continue-message" data-node-id="${nodeId}" title="Continue"${lastAttrs}>${icons.continueGen}</button>
        <button class="message-action-btn message-action-overflow" data-action="actions-overflow" data-node-id="${nodeId}" aria-haspopup="menu" aria-expanded="false" title="More actions">${icons.more}</button>`;
  }

  /** Build the labeled row HTML for one overflow menu item. */
  _buildOverflowRow(nodeId, opts) {
    const { action, className, label, icon, dataAttrs } = opts;
    const cls = `action-overflow-row${className ? ' ' + className : ''}`;
    const dataAttrStr = Object.entries(dataAttrs || {})
      .map(([k, v]) => `data-${k}="${escapeHtml(String(v))}"`)
      .join(' ');
    const actionAttr = action ? `data-action="${action}"` : '';
    return `<button type="button" class="${cls}" data-node-id="${nodeId}" ${actionAttr} ${dataAttrStr}>
      <span class="action-overflow-icon">${icon}</span>
      <span class="action-overflow-label">${escapeHtml(label)}</span>
    </button>`;
  }

  /** Open the overflow popover anchored to the trigger button. Only one
   *  popover is alive at a time — opening a second closes the first.
   *  All popover buttons reuse the same class/data-action selectors as
   *  the legacy inline buttons, so existing delegated handlers fire
   *  identically. Side effect on the trigger: aria-expanded flips. */
  _openActionOverflow(triggerBtn, nodeId, opts = {}) {
    this._closeActionOverflow();

    const isNarrativeSession = !!opts.isNarrativeSession;
    const isLastAssistant = opts.isLastAssistant !== false;
    // Context-gating: only surface rows whose feature actually applies
    // right now. Previously the rows were unconditional and clicking an
    // inapplicable one fired a "doesn't work here" shake — which is
    // worse UX than just not showing the row.
    const hasLanguagePartner = !!window.narrative?.activeCharacter?.ttsVoiceLang;

    const rows = [];
    // Read aloud lives in the primary row now (see _buildAssistantActionsInnerHtml).
    if (hasLanguagePartner) {
      rows.push(this._buildOverflowRow(nodeId, {
        action: '', className: 'breakdown-msg-btn', label: 'Break down', icon: icons.breakdown,
        dataAttrs: { 'node-id': nodeId },
      }));
    }
    if (isLastAssistant && isNarrativeSession) {
      rows.push(this._buildOverflowRow(nodeId, {
        action: 'impersonate-message', label: 'Impersonate', icon: icons.impersonate,
      }));
    }
    if (isLastAssistant) {
      rows.push(this._buildOverflowRow(nodeId, {
        action: 'rerun-as-pick', label: 'Rerun in another mode', icon: icons.rerun,
      }));
    }
    if (isNarrativeSession) {
      rows.push(this._buildOverflowRow(nodeId, {
        action: 'graduate-to-becca', label: 'Share with Becca', icon: icons.shareStar,
      }));
    }
    rows.push(`<div class="action-overflow-divider"></div>`);
    rows.push(this._buildOverflowRow(nodeId, {
      action: 'delete-message', className: 'action-overflow-row--danger', label: 'Delete message', icon: icons.trash,
    }));

    const pop = document.createElement('div');
    pop.className = 'action-overflow-popover';
    pop.setAttribute('role', 'menu');
    pop.innerHTML = rows.join('');
    document.body.appendChild(pop);
    // Popover is body-appended (so it can escape overflow-hidden chat
    // chrome), but the delegated action handler lives on `messagesEl`.
    // Without this listener, clicks on row buttons (delete-message,
    // .tts-msg-btn, rerun-as-pick) never reach the handler and the
    // buttons look broken. Listener dies with the element on close.
    pop.addEventListener('click', this._boundClickHandler);
    this._actionOverflow = { popEl: pop, triggerBtn };
    this._anchorPopoverTo(pop, triggerBtn);
    triggerBtn.setAttribute('aria-expanded', 'true');

    this._actionOverflowOutsideHandler = (ev) => {
      if (!this._actionOverflow) return;
      const { popEl, triggerBtn: t } = this._actionOverflow;
      if (popEl.contains(ev.target)) return;
      if (t.contains(ev.target)) return;
      this._closeActionOverflow();
    };
    this._actionOverflowKeyHandler = (ev) => {
      if (ev.key === 'Escape' && this._actionOverflow) {
        ev.preventDefault();
        this._actionOverflow.triggerBtn?.focus();
        this._closeActionOverflow();
      }
    };
    // Defer so the click that opened the popover doesn't immediately close it.
    setTimeout(() => {
      document.addEventListener('click', this._actionOverflowOutsideHandler);
    }, 0);
    document.addEventListener('keydown', this._actionOverflowKeyHandler);
  }

  _closeActionOverflow() {
    if (this._actionOverflow) {
      const { popEl, triggerBtn } = this._actionOverflow;
      if (popEl?.parentNode) popEl.parentNode.removeChild(popEl);
      if (triggerBtn) triggerBtn.setAttribute('aria-expanded', 'false');
      this._actionOverflow = null;
    }
    if (this._actionOverflowOutsideHandler) {
      document.removeEventListener('click', this._actionOverflowOutsideHandler);
      this._actionOverflowOutsideHandler = null;
    }
    if (this._actionOverflowKeyHandler) {
      document.removeEventListener('keydown', this._actionOverflowKeyHandler);
      this._actionOverflowKeyHandler = null;
    }
  }

  /** Anchor a popover element to a trigger button. Below if there's room,
   *  above otherwise. Right-aligned by default, falling back to left-aligned
   *  if right-alignment would clip the viewport edge. Returns nothing — caller
   *  owns the popover element. */
  _anchorPopoverTo(popEl, triggerBtn) {
    const rect = triggerBtn.getBoundingClientRect();
    const popRect = popEl.getBoundingClientRect();
    const spaceBelow = window.innerHeight - rect.bottom;
    const popHeight = popRect.height || 280;
    const placeAbove = spaceBelow < popHeight + 12 && rect.top > popHeight + 12;
    popEl.style.position = 'fixed';
    if (placeAbove) {
      popEl.style.top = `${Math.max(8, rect.top - popHeight - 6)}px`;
    } else {
      popEl.style.top = `${rect.bottom + 6}px`;
    }
    const popWidth = popRect.width || 240;
    const right = Math.max(8, window.innerWidth - rect.right);
    popEl.style.right = `${right}px`;
    popEl.style.left = 'auto';
    if (rect.right - popWidth < 8) {
      popEl.style.right = 'auto';
      popEl.style.left = '8px';
    }
  }

  /** Open the inline mode picker. Replaces the action overflow popover —
   *  same anchor, same outside-handler treatment, so the user perceives a
   *  smooth drill-down rather than two stacked menus. Selecting a mode
   *  fires `_onAction('rerun-as', nodeId, { content, targetMode })`, which
   *  index.js consumes (no more window.prompt). */
  _openModePickerPopover(triggerBtn, nodeId, currentMode, content) {
    this._closeActionOverflow();

    const modes = [
      { id: 'passthrough', label: 'Passthrough', sub: 'plain chat' },
      { id: 'narrative',   label: 'Narrative',   sub: 'character / story' },
      { id: 'analytical',  label: 'Analytical',  sub: 'reasoning flow' },
      { id: 'agentic',     label: 'Agentic',     sub: 'tool use' },
      { id: 'coder',       label: 'Coder',       sub: 'code workspace' },
    ].filter(m => m.id !== currentMode);

    const rows = modes.map(m => `
      <button type="button" class="action-overflow-row mode-picker-row"
              data-action="rerun-as-go" data-node-id="${nodeId}" data-target-mode="${m.id}">
        <span class="action-overflow-label">
          <span class="mode-picker-name">${escapeHtml(m.label)}</span>
          <span class="mode-picker-sub">${escapeHtml(m.sub)}</span>
        </span>
      </button>
    `).join('');

    const pop = document.createElement('div');
    pop.className = 'action-overflow-popover action-overflow-popover--picker';
    pop.setAttribute('role', 'menu');
    pop.dataset.rerunContent = content || '';
    pop.innerHTML = `<div class="action-overflow-header">Rerun as…</div>${rows}`;
    document.body.appendChild(pop);
    // Same reason as in _openActionOverflow — popover is body-appended so
    // the messagesEl-bound delegated handler doesn't see clicks. Wire the
    // handler directly on the popover so `data-action="rerun-as-go"` fires.
    pop.addEventListener('click', this._boundClickHandler);
    this._actionOverflow = { popEl: pop, triggerBtn };
    this._anchorPopoverTo(pop, triggerBtn);

    triggerBtn.setAttribute('aria-expanded', 'true');

    this._actionOverflowOutsideHandler = (ev) => {
      if (!this._actionOverflow) return;
      const { popEl: p, triggerBtn: t } = this._actionOverflow;
      if (p.contains(ev.target)) return;
      if (t.contains(ev.target)) return;
      this._closeActionOverflow();
    };
    this._actionOverflowKeyHandler = (ev) => {
      if (ev.key === 'Escape' && this._actionOverflow) {
        ev.preventDefault();
        this._actionOverflow.triggerBtn?.focus();
        this._closeActionOverflow();
      }
    };
    setTimeout(() => {
      document.addEventListener('click', this._actionOverflowOutsideHandler);
    }, 0);
    document.addEventListener('keydown', this._actionOverflowKeyHandler);
  }

  /** Same as above, wrapped in the .message-actions div. */
  _buildAssistantActionsHtml(nodeId, opts = {}) {
    return `<div class="message-actions">${this._buildAssistantActionsInnerHtml(nodeId, opts)}</div>`;
  }

  // -----------------------------------------------------------------------
  // Single message element
  // -----------------------------------------------------------------------

  /**
   * Create a DOM element for a single chat message node.
   * @param {object} node    - Tree node
   * @param {object} session - Session object
   * @returns {HTMLElement}
   */
  createMessageEl(node, session) {
    const msg = document.createElement('div');
    msg.className = `message message-${node.role}`;
    if (node.isGreeting) msg.classList.add('greeting-message');
    msg.dataset.nodeId = node.id;

    const isUser = node.role === 'user';
    const isNarrativeSession = session.mode === 'narrative';
    const avatarInnerHtml = _buildAvatarInnerHtml(node.role, session);
    if (!avatarInnerHtml) msg.classList.add('message--no-avatar');

    // Branch navigation
    const siblingInfo = getSiblingInfo(session, node.id);
    let branchHtml = '';
    if (siblingInfo) {
      branchHtml = `
        <div class="swipe-controls">
          <button class="swipe-btn" data-action="branch-prev" data-node-id="${node.id}" ${siblingInfo.index === 0 ? 'disabled' : ''}>${icons.chevronLeft}</button>
          <span class="swipe-indicator">${siblingInfo.index + 1}/${siblingInfo.total}</span>
          <button class="swipe-btn" data-action="branch-next" data-node-id="${node.id}" ${siblingInfo.index === siblingInfo.total - 1 ? 'disabled' : ''}>${icons.chevronRight}</button>
        </div>`;
    }

    // Thinking block (persisted reasoning). Rebuilds from the saved
    // node.reasoning, which may contain UARF phases, model-native thinking,
    // or both. Without either we skip the block.
    let thinkingHtml = '';
    if (!isUser && node.reasoning && ((node.reasoning.phases && node.reasoning.phases.length > 0) || node.reasoning.thinking)) {
      thinkingHtml = this._buildStoredThinkingHtml(node.reasoning);
    }

    // Actions
    let actionsHtml = '';
    if (isUser) {
      actionsHtml = `
        <div class="message-actions">
          <button class="message-action-btn" data-action="edit-message" data-node-id="${node.id}" title="Edit">${icons.edit}</button>
          <button class="message-action-btn" data-action="delete-message" data-node-id="${node.id}" title="Delete">${icons.trash}</button>
        </div>`;
    } else {
      const path = getActivePath(session);
      const lastAssistantNode = [...path].reverse().find(n => n.role === 'assistant');
      const isLastAssistant = lastAssistantNode && lastAssistantNode.id === node.id;
      actionsHtml = this._buildAssistantActionsHtml(node.id, {
        isLastAssistant: !!isLastAssistant,
        isNarrativeSession,
      });

      // A/B test voting buttons
      if (node.ab_test) {
        const ab = node.ab_test;
        const voted = node.ab_vote || '';
        actionsHtml += `
        <div class="ab-test-vote" data-balancer-id="${escapeHtml(ab.balancer_id)}" data-model="${escapeHtml(ab.model_used)}" data-backend="${escapeHtml(ab.backend_key)}" data-node-id="${node.id}">
          <span class="ab-model-label">${escapeHtml(ab.model_used)}@${escapeHtml(ab.backend_key)}</span>
          <button class="ab-vote-btn${voted === 'up' ? ' voted' : ''}" data-vote="up" title="Good response">\ud83d\udc4d</button>
          <button class="ab-vote-btn${voted === 'down' ? ' voted' : ''}" data-vote="down" title="Poor response">\ud83d\udc4e</button>
        </div>`;
      }
    }

    // Inline image thumbnails for VL messages. Each image is wrapped
    // in a cast-btn-host span so the hover-revealed Cast button (top-
    // right overlay) shows when the user hovers the thumbnail without
    // adding visual weight when they aren't aiming at it.
    let imagesHtml = '';
    if (node.images && node.images.length > 0) {
      const realImages = node.images.filter(src => src && src !== '[image]');
      const placeholderCount = node.images.length - realImages.length;
      if (realImages.length > 0) {
        imagesHtml = `<div class="message-images">${realImages.map(src => `
          <span class="message-image-host cast-btn-host">
            <img src="${escapeHtml(src)}" alt="Attached image" class="message-image-thumb md-image" loading="lazy" decoding="async" onerror="this.style.display='none'" />
            <button type="button" class="cast-btn cast-btn-sm cast-btn-on-image cast-btn-hover-reveal message-image-cast"
                    data-action="cast-image" data-src="${escapeHtml(src)}"
                    title="Cast to TV" aria-label="Cast to TV">
              <span class="cast-btn-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M2 16.1A5 5 0 0 1 5.9 20"/>
                  <path d="M2 12.05A9 9 0 0 1 9.95 20"/>
                  <path d="M2 8V6a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-6"/>
                  <line x1="2" y1="20" x2="2.01" y2="20"/>
                </svg>
              </span>
            </button>
          </span>
        `).join('')}</div>`;
      } else if (placeholderCount > 0) {
        imagesHtml = `<div class="message-images"><span class="image-placeholder">${placeholderCount} image${placeholderCount > 1 ? 's' : ''} attached</span></div>`;
      }
    }

    // Character name header for narrative assistant messages
    let nameHeaderHtml = '';
    if (isNarrativeSession && !isUser && session.title) {
      nameHeaderHtml = `<div class="message-char-name">${escapeHtml(session.title)}</div>`;
    }

    // Phase 8 — peer-served badge. When the model that produced this
    // turn appears in the cached model list with a peer_icon (fabric-
    // peer-hosted), render the icon adjacent to the role label so the
    // operator can retroactively tell where the inference ran. Pure
    // client-side derivation; no protocol change.
    let peerBadgeHtml = '';
    if (!isUser && node.model_used) {
      try {
        const cached = (window.__augmentumCachedModels) || [];
        const match = cached.find((m) => m && m.name === node.model_used);
        if (match) {
          const peerIcon = (match.details && match.details.augmentum_peer_icon)
            || (match.augmentum_peer && match.augmentum_peer.icon)
            || '';
          if (peerIcon) {
            const peerHost = (match.details && match.details.augmentum_peer_hostname)
              || (match.augmentum_peer && match.augmentum_peer.hostname)
              || 'a fabric peer';
            peerBadgeHtml = `<span class="message-peer-badge" title="Served by ${escapeHtml(peerHost)}">${escapeHtml(peerIcon)}</span>`;
          }
        }
      } catch (_) { /* defensive: never block render on a lookup */ }
    }

    // Multi-model fan-out — tag which model produced this sibling so the
    // user can tell compare alternatives apart when swiping branches.
    let modelChipHtml = '';
    if (!isUser && node.multi_model && node.model_used) {
      modelChipHtml = `<span class="mm-model-chip" title="Generated by ${escapeHtml(node.model_used)}">${escapeHtml(node.model_used)}</span>`;
    }

    // Generation stats metadata (TTFT, tok/s, token counts, context usage)
    let speedHtml = '';
    if (!isUser) {
      const statsArg = {
        tps: node.tokens_per_second || 0,
        promptTokens: node.prompt_tokens || 0,
        promptTokensEvaluated: node.prompt_tokens_evaluated || 0,
        promptTokensCached: node.prompt_tokens_cached || 0,
        promptTokensCacheWrite: node.prompt_tokens_cache_write || 0,
        kvReuse: node.kv_reuse || '',
        kvVoidCause: node.kv_void_cause || '',
        reasoningTokens: node.reasoning_tokens || 0,
        evalTokens: node.eval_tokens || 0,
        contextLen: node.context_length || 0,
        contextUsed: node.context_used || 0,
        ttftMs: node.ttft_ms || 0,
        totalDurationMs: node.total_duration_ms || 0,
        evalDurationMs: node.eval_duration_ms || 0,
        promptTokensEstimated: !!node.prompt_tokens_estimated,
        evalTokensEstimated: !!node.eval_tokens_estimated,
      };
      const parts = _buildGenerationStatParts(statsArg);
      if (parts.length > 0) {
        speedHtml = `<div class="message-gen-speed" title="${escapeHtml(_generationStatsTitle(statsArg))}">${parts.join(' \u00B7 ')}</div>`;
      }
    }

    // Knowledge-pack chip \u2014 shown when retrieval was attempted for this turn
    // (binding present + mode enabled). Distinguishes "found and used" from
    // "found but couldn't fit" so users can tell whether the response is
    // grounded in pack content or the model's own training. Honors the
    // no-silent-truncation rule: an "all_dropped" outcome surfaces visibly
    // rather than the previous behavior of returning generic content with
    // no signal that retrieval ran.
    let packHtml = '';
    if (!isUser && node.knowledgePack) {
      packHtml = _renderKnowledgePackChip(node.knowledgePack);
    }

    // Structural interrupted badge \u2014 replaces the prior `*(incomplete)*`
    // markdown injection. Set in index.js onError when the stream was cut
    // short by a network drop or a backend error chunk. error_message is
    // surfaced via title= for the curious; the body text stays short.
    const interruptedHtml = (!isUser && node.interrupted)
      ? `<div class="interrupted-badge" role="status"${node.error_message ? ` title="${escapeHtml(node.error_message)}"` : ''}>Response interrupted</div>`
      : '';

    msg.innerHTML = `
      <div class="message-avatar">${avatarInnerHtml}</div>
      <div class="message-bubble">
        ${nameHeaderHtml}${peerBadgeHtml}${modelChipHtml}
        ${imagesHtml}
        <div class="message-content" dir="auto">${thinkingHtml}${renderMarkdown(node.content, { mode: this._mode, narrativePanelsCollapsed: this._narrativePanelsCollapsed })}</div>
        ${interruptedHtml}
        ${packHtml}
        ${speedHtml}
        ${branchHtml}
        ${actionsHtml}
      </div>
    `;

    // Copy/TTS handled via delegated click handler — no per-element listeners.
    // Content stored as data attribute for delegation access.
    msg.dataset.rawContent = node.content;

    if (node.agenticArtifactCards && node.agenticArtifactCards.length > 0) {
      this._renderStoredToolCards(msg, node.agenticArtifactCards);
    }

    // Project card
    if (node.projectArtifact) {
      const contentEl = msg.querySelector('.message-content');
      if (contentEl) this._onAction('render-project-card', node.id, { node, contentEl });
    }

    // YouTube discovery cards
    if (node.youtubeData) {
      this._onAction('render-youtube', node.id, { data: node.youtubeData, messageEl: msg });
    }

    // Unified tool cards (image_search gallery, browse_fetch, etc.)
    if (Array.isArray(node.toolCards) && node.toolCards.length) {
      this.replayToolCards(node.toolCards, msg);
    }

    // Narrative internal-tool activity (recall + lorebook) — collapsed chip.
    if (Array.isArray(node.narrativeActivity) && node.narrativeActivity.length) {
      this.replayNarrativeActivity(node.narrativeActivity, msg);
    }

    // World-system event cards (rolls / tracker shifts / sheets)
    if (Array.isArray(node.world_events) && node.world_events.length) {
      import('./world-panel.js')
        .then(m => m.renderWorldEventCards(node.world_events, msg))
        .catch(() => {});
    }

    return msg;
  }

  // -----------------------------------------------------------------------
  // Streaming — create / append / update / finalize
  // -----------------------------------------------------------------------

  /**
   * Create and append a streaming message element.
   * @param {object} session - Current session (for avatar/name)
   * @returns {HTMLElement} The streaming message element
   */
  createStreamingMessage(session) {
    const avatarInnerHtml = _buildAvatarInnerHtml('assistant', session);
    const msg = document.createElement('div');
    msg.className = 'message message-assistant';
    if (!avatarInnerHtml) msg.classList.add('message--no-avatar');
    msg.dataset.nodeId = 'pending';

    const isNarrativeStream = session?.mode === 'narrative';
    const streamNameHtml = (isNarrativeStream && session?.title)
      ? `<div class="message-char-name">${escapeHtml(session.title)}</div>` : '';

    msg.innerHTML = `
      <div class="message-avatar">${avatarInnerHtml}</div>
      <div class="message-bubble">
        ${streamNameHtml}
        <div class="message-content" dir="auto">
          <div class="streaming-dots">
            <span></span><span></span><span></span>
            <span class="streaming-status-label">Processing prompt\u2026</span>
          </div>
        </div>
      </div>
    `;

    // Reset all stream state
    this.resetStreamState();

    // Store as instance property — no DOM id needed
    this.streamingEl = msg;
    this.messagesEl.appendChild(msg);
    return msg;
  }

  /**
   * Resume streaming INTO an existing assistant message bubble.
   *
   * Used by the Continue button — instead of creating a fresh message,
   * we re-mark the existing one as the streaming target. The dataset's
   * raw content is preserved so subsequent ``appendToStreaming`` calls
   * concatenate onto the prior partial, and ``finalizeStreaming``
   * writes the merged result back into the bubble.
   *
   * Strips the post-finalize chrome (actions row, swipe controls,
   * knowledge-pack chip, etc.) so the bubble looks like it's mid-stream
   * again — finalizeStreaming will rebuild that chrome when the
   * continuation completes.
   *
   * Returns the message element on success, ``null`` if the node
   * couldn't be found (caller should fall back to createStreamingMessage).
   *
   * @param {string} nodeId - The assistant node to resume
   * @returns {HTMLElement|null}
   */
  resumeStreamingMessage(nodeId) {
    if (!nodeId || !this.messagesEl) return null;
    const el = this.messagesEl.querySelector(`[data-node-id="${CSS.escape(nodeId)}"]`);
    if (!el || !el.classList.contains('message-assistant')) return null;

    // Strip post-finalize chrome — actions, swipe indicator, knowledge
    // chip, interrupted badge. Stream append will rebuild as content
    // lands; finalize will reattach actions/swipe at the end.
    el.querySelectorAll('.message-actions, .swipe-controls, .knowledge-pack-chip, .interrupted-badge').forEach(n => n.remove());

    const raw = el.dataset.rawContent || '';
    el.dataset.rawContent = raw;

    // Reset stream state BEFORE re-rendering so the stable/active split
    // starts from a clean boundary. The body is re-built from raw via the
    // shared render path.
    this.resetStreamState();
    this.streamingEl = el;

    const content = el.querySelector('.message-content');
    if (content) {
      const responseBody = content.querySelector('.response-body');
      if (responseBody) responseBody.remove();
      this._renderWithStableSplit(content, raw);
    }
    return el;
  }

  /**
   * Append streamed text to the current streaming message.
   *
   * The raw text is accumulated synchronously so abort/finalize always see
   * the latest content, but the DOM re-render is coalesced to one pass per
   * animation frame. Many NDJSON deltas typically arrive per frame; without
   * coalescing we'd full-render the bubble + re-hljs every code block on
   * each one.
   *
   * Thinking-panel auto-collapse on first content was REMOVED here —
   * interleaved-reasoning models (GPT-5, recent Qwen) lost the live
   * reveal. finalizeStreaming still collapses on stream end.
   *
   * @param {string} text - Delta text
   */
  appendToStreaming(text) {
    const el = this.streamingEl;
    if (!el) return;

    // Accumulate immediately so abort/finalize/getStreamingRawContent see
    // the latest, even if no render has flushed yet this frame.
    const current = el.dataset.rawContent || '';
    el.dataset.rawContent = current + text;

    // Forward the delta to the avatar presence engine so the VRMA picker
    // + breathing modulation + gesture cadence track what's being said.
    // Voice mode wires this at voice.js — without this site, text-only
    // chat streams leave the companion looking idle/unaware while she
    // speaks. _avatarOnDelta is sync + allocation-light per chunk; the
    // dynamic import fires once and caches.
    _avatarOnDelta(text);

    // Once-per-stream signal so other subsystems can relax expensive
    // periodic work (avatar viewer's snapshot readback today; more
    // consumers as the bus grows). activity-bus has zero render deps,
    // so it imports statically without pulling Three.js into chat.
    if (!this._streamSignaled) {
      this._streamSignaled = true;
      bus.set('chat_streaming', true);
      // First prose token — collapse the narrative activity strip into its
      // done chip, mirroring how the thinking block settles once the reply
      // begins. Safe for narrative (unlike the reverted thinking auto-
      // collapse): its tools always run BEFORE prose, never interleaved.
      this.collapseNarrativeActivity();
    }

    if (this._streamRenderScheduled) return;
    this._streamRenderScheduled = true;
    const flush = () => {
      this._streamRenderScheduled = false;
      this._renderStreamingBubble();
    };
    if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(flush);
    } else {
      // Test environments / non-browser hosts — flush synchronously.
      flush();
    }
  }

  /** Render the streaming bubble's response body from the current
   *  ``dataset.rawContent``, using the stable/active split so we never
   *  re-render or re-highlight content that's already settled. Private —
   *  called from the rAF flush in ``appendToStreaming`` and from
   *  ``replaceStreamedContent``/``resumeStreamingMessage`` to share one
   *  render pipeline. */
  _renderStreamingBubble() {
    const el = this.streamingEl;
    if (!el) return;
    const content = el.querySelector('.message-content');
    if (!content) return;

    // Remove dots once content arrives
    const dots = content.querySelector('.streaming-dots');
    if (dots) dots.remove();

    // Strip the hidden [motion:xxx] avatar tag (incl. a half-streamed one) so
    // it never flashes in the bubble. The cue is read + dispatched at finalize.
    const raw = stripMotionCueStreaming(el.dataset.rawContent || '');
    this._renderWithStableSplit(content, raw);
    // Instant scroll during streaming: per-frame bottom-delta is a few
    // pixels, so smooth-scroll has no visible benefit but DOES schedule a
    // per-frame compositor animation. Other scrollToBottom call sites
    // (turn boundaries, finalize, viewport resize) stay smooth.
    this.scrollToBottom(false);
  }

  /** @private
   *  Render ``raw`` into ``content``'s ``.response-body`` using a two-part
   *  split:
   *    - ``.stream-stable``  — text up to the most-recent safe boundary
   *      (paragraph break outside any open code/panel fence). Rendered once
   *      per promotion; hljs runs only when the code-block count grew.
   *    - ``.stream-active``  — trailing suffix that the model is still
   *      writing. Re-rendered every frame; hljs runs only if it contains
   *      a code block (suffix is short — cheap).
   *
   *  Both children use ``display: contents`` (CSS) so their rendered
   *  markdown promotes into ``.response-body``'s block flow without
   *  introducing visible structural boundaries. The streaming cursor
   *  pseudo-element on ``.response-body`` therefore appears in the same
   *  visual position as the pre-split single-block render did. */
  _renderWithStableSplit(content, raw) {
    // Shared incremental renderer (chat/stream-render.js): stable/active
    // split + append-only open-fence handling + deferred highlight. Same
    // path the coder conversation view uses — see that module's header for
    // why the old per-frame full re-parse + re-highlight was O(n²).
    renderStreamSplit(content, raw, this._streamSplit, {
      mode: this._mode,
      narrativePanelsCollapsed: this._narrativePanelsCollapsed,
      highlightHooks: this._highlightHooks,
    });
  }

  /**
   * Update the UARF/analytical reasoning phases display in the streaming message.
   * @param {Array} phases     - Phase objects with { name, status }
   * @param {string} complexity - Complexity label
   */
  updateStreamThinking(phases, complexity) {
    const el = this.streamingEl;
    if (!el) return;
    const content = el.querySelector('.message-content');

    const dots = content.querySelector('.streaming-dots');
    if (dots) dots.remove();

    this._streamPhases     = phases || this._streamPhases;
    this._streamComplexity = complexity || this._streamComplexity;

    // Clear content for re-running phases (backtracking)
    for (const p of this._streamPhases) {
      if (p.status === 'running') {
        this._streamPhaseContent[p.name] = '';
      }
    }

    // First-render flag — opens the panel by default for live streams.
    if (!content.querySelector('.reasoning-summary')) {
      this._streamThinkingOpen = true;
    }
    // Re-render through the shared builder so the live UARF surface uses
    // identical markup to the stored variant (one source of markup truth;
    // historic drift between live/stored is what this refactor fixes).
    const panelHtml = this._buildPhasesPanelHtml({
      phases:       this._streamPhases,
      toolCalls:    this._streamToolCalls,
      phaseContent: this._streamPhaseContent,
      flowName:     this._streamFlowName,
    }, {
      variant:     'live',
      summaryOpen: this._streamThinkingOpen,
    });
    const existing = content.querySelector('.reasoning-summary');
    if (existing) {
      existing.outerHTML = panelHtml;
    } else {
      const tmp = document.createElement('div');
      tmp.innerHTML = panelHtml;
      const node = tmp.firstElementChild;
      if (node) content.insertBefore(node, content.firstChild);
    }
    // Instant scroll during streaming — smooth here re-armed a compositor
    // animation on every phase tick (matches the content/thinking paths).
    this.scrollToBottom(false);
  }

  /**
   * Append model thinking (extended thinking / reasoning tokens) to the
   * streaming message.
   *
   * Accumulates synchronously (finalize/getStreamingData need the latest) but
   * coalesces the DOM write + scroll to one pass per animation frame — the
   * same discipline as ``appendToStreaming``. Reasoning streams are large
   * (DeepSeek V4 Pro CoT is ~4× legacy volume) and arrive at full token rate;
   * the previous per-delta ``pre.textContent = whole-string`` + smooth scroll
   * was O(n²) plus a compositor animation per token — the observed freeze.
   *
   * @param {string} delta - Text delta
   */
  appendModelThinking(delta) {
    const el = this.streamingEl;
    if (!el) return;

    this._streamThinking += delta;

    if (this._streamThinkRenderScheduled) return;
    this._streamThinkRenderScheduled = true;
    const flush = () => this._flushModelThinking();
    if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(flush);
    } else {
      flush();
    }
  }

  /** @private
   *  Flush accumulated ``_streamThinking`` into the live thinking block.
   *  First flush builds the block (shared builder → identical markup to the
   *  stored variant). Subsequent flushes append ONLY the new tail as a text
   *  node — O(delta), no full-string rewrite, no chrome re-paint. The rendered
   *  length rides on the block element so resume/rebuild from another path
   *  resyncs safely instead of duplicating. Scroll is instant + coalesced into
   *  this frame, matching the content path. */
  _flushModelThinking() {
    this._streamThinkRenderScheduled = false;
    const el = this.streamingEl;
    if (!el) return;
    const content = el.querySelector('.message-content');
    if (!content) return;

    const dots = content.querySelector('.streaming-dots');
    if (dots) dots.remove();

    let block = content.querySelector('.model-thinking-block');
    if (!block) {
      const html = this._buildModelThinkingHtml(this._streamThinking, {
        variant: 'live',
        open: true,
      });
      const tmp = document.createElement('div');
      tmp.innerHTML = html;
      block = tmp.firstElementChild;
      if (!block) return;
      const uarfBlock = content.querySelector('.thinking-block:not(.model-thinking-block)');
      if (uarfBlock) {
        content.insertBefore(block, uarfBlock);
      } else {
        content.insertBefore(block, content.firstChild);
      }
      // The builder already rendered the full accumulated text.
      block._thinkRenderedLen = this._streamThinking.length;
    } else {
      const pre = block.querySelector('.model-thinking-pre');
      if (pre) {
        const rendered = typeof block._thinkRenderedLen === 'number' ? block._thinkRenderedLen : -1;
        const plan = planThinkingAppend(rendered, this._streamThinking.length);
        if (plan.action === 'resync') {
          // Block came from a different path (stored rebuild / resume) or the
          // accumulator was reset under it — resync wholesale once, O(n).
          pre.textContent = this._streamThinking;
        } else if (plan.action === 'append') {
          // Common path: append only the unrendered tail. O(delta).
          pre.appendChild(document.createTextNode(this._streamThinking.slice(plan.from)));
        }
        block._thinkRenderedLen = this._streamThinking.length;
      }
    }
    this.scrollToBottom(false);
  }

  /**
   * Update the streaming status label text.
   * @param {string} stage - Status key or text
   */
  updateStreamingStatus(stage) {
    const el = this.streamingEl;
    if (!el) return;
    const label = el.querySelector('.streaming-status-label');
    if (label) label.textContent = _STATUS_LABELS[stage] || stage;
  }

  /**
   * Update the prefill-progress bar that sits next to the streaming
   * status label. Called by the poll loop in chat/prefill-progress.js
   * while llama-server is processing the prompt before first token.
   * Lazily injects the bar DOM the first time progress arrives so the
   * "Loading model…" and other stages don't show an empty bar.
   *
   * `progress` is 0..1; tokens_done/tps/elapsed_s are display-only.
   */
  setPrefillProgress({ progress, tokens_done, tps, elapsed_s }) {
    const el = this.streamingEl;
    if (!el) return;
    const dots = el.querySelector('.streaming-dots');
    if (!dots) return;
    let bar = dots.querySelector('.streaming-prefill-bar');
    if (!bar) {
      bar = document.createElement('div');
      bar.className = 'streaming-prefill-bar';
      bar.innerHTML = '<div class="streaming-prefill-fill"></div>';
      dots.appendChild(bar);
    }
    const pct = Math.max(0, Math.min(100, Math.round(progress * 100)));
    const fill = bar.querySelector('.streaming-prefill-fill');
    if (fill) fill.style.width = `${pct}%`;
    const label = el.querySelector('.streaming-status-label');
    if (label) {
      // Compact one-liner that fits in the bubble. tps + remaining
      // elapsed give the user a rough ETA without a separate ETA field
      // — e.g. "Preparing context · 47% · 96 tok/s".
      const tpsStr = tps > 0 ? ` · ${Math.round(tps)} tok/s` : '';
      label.textContent = `Preparing context · ${pct}%${tpsStr}`;
    }
  }

  /**
   * Render or clear an inline stall affordance inside the streaming bubble.
   * Two states:
   *   - 'thinking': soft hint after 4s of no content (likely just slow first-token)
   *   - 'stalled':  louder banner after 15s — backend or model genuinely unresponsive
   * The 'stalled' variant includes an Abort & retry button that fires the
   * `augmentum:stall-abort-retry` event; index.js owns the actual recovery.
   * @param {'none'|'thinking'|'stalled'} state
   */
  setStallBanner(state) {
    const el = this.streamingEl;
    if (!el) return;
    const bubble = el.querySelector('.message-bubble');
    if (!bubble) return;

    let banner = bubble.querySelector('.stall-banner');

    if (state === 'none') {
      if (banner) banner.remove();
      return;
    }

    if (!banner) {
      banner = document.createElement('div');
      banner.className = 'stall-banner';
      // Insert before any action row / chrome that may have been added.
      const beforeNode = bubble.querySelector('.message-actions')
        || bubble.querySelector('.swipe-controls')
        || null;
      bubble.insertBefore(banner, beforeNode);
    }

    banner.classList.remove('stall-banner--thinking', 'stall-banner--stalled');
    banner.classList.add(`stall-banner--${state}`);

    if (state === 'thinking') {
      banner.innerHTML = `<span class="stall-banner-text">Just a moment…</span>`;
    } else {
      // Friendlier register — the previous "Stream stalled. The model
      // or backend isn't responding. [Abort & retry]" copy read as a
      // crash. This phrasing tells the user we're still on it but
      // offers a soft escape hatch.
      banner.innerHTML = `
        <span class="stall-banner-text">Still here — just slower than usual.</span>
        <button type="button" class="stall-banner-btn" data-action="stall-abort-retry">Try again</button>
      `;
    }
  }

  /** Tear down the prefill bar so the next stage's label is uncluttered. */
  clearPrefillProgress() {
    const el = this.streamingEl;
    if (!el) return;
    const bar = el.querySelector('.streaming-prefill-bar');
    if (bar) bar.remove();
  }

  /**
   * Show a soft progress bar for the current model_load / model_swap stage.
   * Called by the load-progress poll loop. The bar uses the same DOM
   * shape as the prefill bar (so the CSS rules already cover layout)
   * but is rendered into a distinct class so the two can't visually
   * collide if a swap is immediately followed by a prefill.
   *
   * `progress` is 0..0.95 (the backend caps to avoid claiming 100%
   * before the manager actually reaches READY). elapsed_s + expected_s
   * are display-only — turned into a "14s of ~30s" suffix so the user
   * gets an honest ETA rather than a bare percentage.
   */
  setLoadProgress({ model_id, progress, elapsed_s, expected_s, stage_label }) {
    const el = this.streamingEl;
    if (!el) return;
    const dots = el.querySelector('.streaming-dots');
    if (!dots) return;
    let bar = dots.querySelector('.streaming-load-bar');
    if (!bar) {
      bar = document.createElement('div');
      bar.className = 'streaming-prefill-bar streaming-load-bar';
      bar.innerHTML = '<div class="streaming-prefill-fill"></div>';
      dots.appendChild(bar);
    }
    const pct = Math.max(0, Math.min(100, Math.round((progress ?? 0) * 100)));
    const fill = bar.querySelector('.streaming-prefill-fill');
    if (fill) fill.style.width = `${pct}%`;
    const label = el.querySelector('.streaming-status-label');
    if (label) {
      const head = stage_label || 'Loading model';
      const name = model_id ? ` · ${model_id}` : '';
      const elapsed = Math.max(0, Math.round(elapsed_s || 0));
      const expected = Math.max(0, Math.round(expected_s || 0));
      // "Loading model · deepseek-v3 · 14s of ~30s" — three slots,
      // the last one is the ETA suffix when we have an expected.
      const timing = expected > 0
        ? ` · ${elapsed}s of ~${expected}s`
        : ` · ${elapsed}s`;
      label.textContent = `${head}${name}${timing}`;
    }
  }

  /** Tear down the load bar so the next stage's label is uncluttered. */
  clearLoadProgress() {
    const el = this.streamingEl;
    if (!el) return;
    const bar = el.querySelector('.streaming-load-bar');
    if (bar) bar.remove();
  }

  /**
   * Add a tool call to the stream state.
   * @param {object} toolCall - Tool call metadata
   */
  addStreamToolCall(toolCall) {
    this._streamToolCalls.push(toolCall);
    // If the tool returned a structured card, render it inline in the
    // streaming bubble so the user gets a clean card with preview / edit /
    // download instead of the model's terse follow-up text.
    if (toolCall && toolCall.card && toolCall.success !== false) {
      this._renderInlineToolCard(toolCall.card);
    }
  }

  renderStreamingToolCard(card) {
    this._renderInlineToolCard(card);
  }

  _ensureToolCardHost(content) {
    if (!content) return null;
    let host = content.querySelector('.tool-card-host');
    if (!host) {
      host = document.createElement('div');
      host.className = 'tool-card-host';
      const responseBody = content.querySelector('.response-body');
      if (responseBody && responseBody.parentElement === content) {
        content.insertBefore(host, responseBody);
      } else {
        content.appendChild(host);
      }
    }
    return host;
  }

  _renderInlineToolCard(card) {
    const el = this.streamingEl;
    if (!el) return;
    const content = el.querySelector('.message-content');
    const host = this._ensureToolCardHost(content);
    if (!host) return;
    import('./tool-card.js').then(m => {
      const artifactId = String(card?.artifact_id || '');
      if (artifactId && host.querySelector(`.tool-card--typed[data-artifact-id="${CSS.escape(artifactId)}"]`)) {
        return;
      }
      const html = m.renderToolCard(card);
      if (!html) return;
      const wrap = document.createElement('div');
      wrap.innerHTML = html;
      if (wrap.firstElementChild) host.appendChild(wrap.firstElementChild);
    }).catch(() => { /* fall back to nothing — text summary still in bubble */ });
  }

  /**
   * Add phase content delta to stream state.
   * @param {string} phaseName
   * @param {string} delta
   */
  addStreamPhaseContent(phaseName, delta) {
    if (!this._streamPhaseContent[phaseName]) this._streamPhaseContent[phaseName] = '';
    this._streamPhaseContent[phaseName] += delta;
  }

  /**
   * Set the flow name displayed in the reasoning summary header.
   * @param {string} name
   */
  setStreamFlowName(name) {
    this._streamFlowName = name;
  }

  /**
   * Update streaming metrics (tok/s, context, etc.).
   * @param {object} metrics
   */
  updateStreamMetrics(metrics) {
    if (metrics.tokens_per_second != null || metrics.tps != null) {
      this._streamMetrics.tps = metrics.tokens_per_second ?? metrics.tps;
    }
    if (metrics.context_length != null || metrics.contextLen != null) {
      this._streamMetrics.contextLen = metrics.context_length ?? metrics.contextLen;
    }
    if (metrics.context_used != null || metrics.contextUsed != null) {
      this._streamMetrics.contextUsed = metrics.context_used ?? metrics.contextUsed;
    }
    // Surface context usage to whoever cares. The composer's context
    // meter is driven via augmentum:turn-stats (which carries the same
    // numbers); this event remains as a generic hook for any other
    // subscriber. Fire only when both numbers are present so listeners
    // can render a complete percent.
    if (
      this._streamMetrics.contextLen > 0
      && this._streamMetrics.contextUsed > 0
    ) {
      document.dispatchEvent(new CustomEvent('augmentum:context-usage', {
        detail: {
          used: this._streamMetrics.contextUsed,
          total: this._streamMetrics.contextLen,
        },
      }));
    }
    if (metrics.prompt_tokens != null || metrics.promptTokens != null) {
      this._streamMetrics.promptTokens = metrics.prompt_tokens ?? metrics.promptTokens;
    }
    if (metrics.prompt_tokens_evaluated != null || metrics.promptTokensEvaluated != null) {
      this._streamMetrics.promptTokensEvaluated = metrics.prompt_tokens_evaluated ?? metrics.promptTokensEvaluated;
    }
    if (metrics.prompt_tokens_cache_write != null || metrics.promptTokensCacheWrite != null) {
      this._streamMetrics.promptTokensCacheWrite = metrics.prompt_tokens_cache_write ?? metrics.promptTokensCacheWrite;
    }
    if (metrics.prompt_tokens_cached != null || metrics.promptTokensCached != null) {
      this._streamMetrics.promptTokensCached = metrics.prompt_tokens_cached ?? metrics.promptTokensCached;
    }
    if (metrics.kv_reuse != null || metrics.kvReuse != null) {
      this._streamMetrics.kvReuse = metrics.kv_reuse ?? metrics.kvReuse;
    }
    if (metrics.kv_void_cause != null || metrics.kvVoidCause != null) {
      this._streamMetrics.kvVoidCause = metrics.kv_void_cause ?? metrics.kvVoidCause;
    }
    if (metrics.eval_tokens != null || metrics.evalTokens != null) {
      this._streamMetrics.evalTokens = metrics.eval_tokens ?? metrics.evalTokens;
    }
    if (metrics.reasoning_tokens != null || metrics.reasoningTokens != null) {
      this._streamMetrics.reasoningTokens = metrics.reasoning_tokens ?? metrics.reasoningTokens;
    }
    if (metrics.ttft_ms != null || metrics.ttftMs != null) {
      this._streamMetrics.ttftMs = metrics.ttft_ms ?? metrics.ttftMs;
    }
    if (metrics.total_duration_ms != null || metrics.totalDurationMs != null) {
      this._streamMetrics.totalDurationMs = metrics.total_duration_ms ?? metrics.totalDurationMs;
    }
    if (metrics.eval_duration_ms != null || metrics.evalDurationMs != null) {
      this._streamMetrics.evalDurationMs = metrics.eval_duration_ms ?? metrics.evalDurationMs;
    }
    if (metrics.prompt_tokens_estimated != null) {
      this._streamMetrics.promptTokensEstimated = !!metrics.prompt_tokens_estimated;
    }
    if (metrics.eval_tokens_estimated != null) {
      this._streamMetrics.evalTokensEstimated = !!metrics.eval_tokens_estimated;
    }

    // Push the merged snapshot to the ambient stats bar above the
    // composer. Fired on every metrics update (including in-flight tps
    // ticks), so the bar tracks live during generation and lands on
    // the final values when the stream terminates. The bar's own
    // listener filters down to renderable parts.
    const m = this._streamMetrics;
    document.dispatchEvent(new CustomEvent('augmentum:turn-stats', {
      detail: {
        tps: m.tps,
        evalTokens: m.evalTokens,
        promptTokens: m.promptTokens,
        promptTokensEvaluated: m.promptTokensEvaluated,
        promptTokensCached: m.promptTokensCached,
        kvReuse: m.kvReuse || '',
        kvVoidCause: m.kvVoidCause || '',
        reasoningTokens: m.reasoningTokens,
        contextLen: m.contextLen,
        contextUsed: m.contextUsed,
        ttftMs: m.ttftMs,
        totalDurationMs: m.totalDurationMs,
        evalDurationMs: m.evalDurationMs,
        promptTokensEstimated: m.promptTokensEstimated,
        evalTokensEstimated: m.evalTokensEstimated,
      },
    }));
  }

  /**
   * Replace streamed raw content (used for regex-transformed text).
   * Resets the stable/active boundary so the new content is fully
   * re-evaluated on the next render flush.
   * @param {string} newContent
   */
  replaceStreamedContent(newContent) {
    const el = this.streamingEl;
    if (!el) return;
    el.dataset.rawContent = newContent;
    const content = el.querySelector('.message-content');
    if (!content) return;
    // Drop any existing split so _renderWithStableSplit rebuilds clean.
    const responseBody = content.querySelector('.response-body');
    if (responseBody) responseBody.remove();
    this._streamSplit = newSplitState();
    this._renderWithStableSplit(content, newContent);
  }

  /**
   * Get a passthrough tool activity container on the current streaming message.
   * Creates one if it doesn't exist.
   * @returns {HTMLElement|null}
   */
  _getToolContainer(targetEl) {
    // targetEl lets replay (re-render of a finalized node) target a
    // specific message bubble instead of the live streaming one.
    const el = targetEl || this.streamingEl;
    if (!el) return null;
    const bubble = el.querySelector('.message-bubble');
    if (!bubble) return null;
    let container = bubble.querySelector('.pt-tool-activity');
    if (!container) {
      container = document.createElement('div');
      container.className = 'pt-tool-activity';
      bubble.insertBefore(container, bubble.firstChild);
    }
    return container;
  }

  /**
   * Show a "running" tool indicator on the streaming message.
   * @param {string[]} toolNames
   */
  showToolIndicator(toolNames) {
    const container = this._getToolContainer();
    if (!container) return;
    const labels = toolNames.map(n => _PT_TOOL_LABELS[n] || n.replace(/_/g, ' '));
    const indicator = document.createElement('div');
    indicator.className = 'pt-tool-indicator running';
    indicator.innerHTML = `<span class="pt-tool-spinner"></span> Using ${escapeHtml(labels.join(', '))}\u2026`;
    container.appendChild(indicator);
    const statusLabel = _TOOL_STATUS_LABELS[toolNames[0]] || `Using ${labels.join(', ')}`;
    this.updateStreamingStatus(statusLabel);
  }

  /**
   * Render a completed tool call result indicator on the streaming message.
   * @param {object} tc - Tool call object with { tool, success }
   */
  renderToolCallResult(tc) {
    const container = this._getToolContainer();
    if (!container) return;
    const label = _PT_TOOL_LABELS[tc.tool] || tc.tool.replace(/_/g, ' ');
    const icon = tc.success ? '\u2713' : '\u2717';
    const cls = tc.success ? 'success' : 'error';

    // Merge consecutive successes of the SAME tool into one pill with a
    // count (legacy passthrough path can finalize several identical calls in
    // a row); failures always get their own line so they're not lost.
    const last = container.lastElementChild;
    if (tc.success && last && last.classList.contains('pt-tool-indicator')
        && last.dataset.toolResultGroup === tc.tool && !last.classList.contains('running')
        && !last.classList.contains('error')) {
      const count = (+last.dataset.toolResultCount || 1) + 1;
      last.dataset.toolResultCount = String(count);
      last.innerHTML = `<span class="pt-tool-icon ${cls}">${icon}</span> ${escapeHtml(label)} <span class="pt-tool-count">${count}</span>`;
      return;
    }

    const running = container.querySelector('.pt-tool-indicator.running:last-child');
    if (running) {
      running.classList.remove('running');
      running.classList.add(cls);
      if (tc.success) running.dataset.toolResultGroup = tc.tool;
      running.innerHTML = `<span class="pt-tool-icon ${cls}">${icon}</span> ${escapeHtml(label)}`;
    } else {
      const el = document.createElement('div');
      el.className = `pt-tool-indicator ${cls}`;
      if (tc.success) el.dataset.toolResultGroup = tc.tool;
      el.innerHTML = `<span class="pt-tool-icon ${cls}">${icon}</span> ${escapeHtml(label)}`;
      container.appendChild(el);
    }
  }

  // --- Agentic chain: one indicator per (tool, wave) -----------------------
  // A parallel wave of N calls to the same tool would otherwise spawn N
  // "Using X…" lines and N "✓ X" lines (chain_step events fire per step).
  // Collapse them: a single indicator per running group, bumped by count.

  /** A chain step for `tool` started. */
  chainToolRunning(tool) {
    const container = this._getToolContainer();
    if (!container || !tool) return;
    let el = container.querySelector(`.pt-tool-indicator.running[data-tool-group="${CSS.escape(tool)}"]`);
    if (el) {
      el.dataset.active = String((+el.dataset.active || 0) + 1);
      el.dataset.total = String((+el.dataset.total || 0) + 1);
    } else {
      el = document.createElement('div');
      el.className = 'pt-tool-indicator running';
      el.dataset.toolGroup = tool;
      el.dataset.active = '1';
      el.dataset.total = '1';
      el.dataset.failed = '0';
      container.appendChild(el);
    }
    this._paintChainRunning(el, tool);
    this.updateStreamingStatus(_chainVerb(tool));
  }

  /** A chain step for `tool` finished (success/failure). */
  chainToolDone(tool, success) {
    const container = this._getToolContainer();
    if (!container || !tool) return;
    const el = container.querySelector(`.pt-tool-indicator.running[data-tool-group="${CSS.escape(tool)}"]`);
    if (!el) {
      // No open group (out-of-order, or another path drew the running pill) —
      // append a one-off completed line so the step isn't lost.
      const cls = success ? 'success' : 'error';
      const one = document.createElement('div');
      one.className = `pt-tool-indicator ${cls}`;
      one.innerHTML = `<span class="pt-tool-icon ${cls}">${success ? '✓' : '✗'}</span> ${escapeHtml(_chainNoun(tool))}`;
      container.appendChild(one);
      return;
    }
    const active = Math.max(0, (+el.dataset.active || 1) - 1);
    el.dataset.active = String(active);
    if (!success) el.dataset.failed = String((+el.dataset.failed || 0) + 1);
    if (active > 0) { this._paintChainRunning(el, tool); return; }
    // Wave complete.
    const total = +el.dataset.total || 1;
    const failed = +el.dataset.failed || 0;
    const ok = failed === 0;
    const cls = ok ? 'success' : 'error';
    el.classList.remove('running');
    el.classList.add(cls);
    el.removeAttribute('data-tool-group');
    const count = total > 1 ? ` <span class="pt-tool-count">${total}</span>` : '';
    const failNote = failed ? ` <span class="pt-tool-count pt-tool-count--fail">${failed} failed</span>` : '';
    el.innerHTML = `<span class="pt-tool-icon ${cls}">${ok ? '✓' : '✗'}</span> ${escapeHtml(_chainNoun(tool))}${count}${failNote}`;
  }

  _paintChainRunning(el, tool) {
    const total = +el.dataset.total || 1;
    const done = total - (+el.dataset.active || 0);
    const badge = total > 1
      ? ` <span class="pt-tool-count">${done > 0 ? `${done}/${total}` : total}</span>`
      : '';
    el.innerHTML = `<span class="pt-tool-spinner"></span> ${escapeHtml(_chainVerb(tool))}…${badge}`;
  }

  /**
   * Render a tool_start event — creates a live tool card with label + context.
   * Keyed by event.id so tool_progress / tool_complete can find and update it.
   * @param {object} ev - { id, tool, label, context?, phase? }
   */
  renderToolStart(ev, targetEl) {
    const container = this._getToolContainer(targetEl);
    if (!container || !ev || !ev.id) return;
    // Capture for persistence on the assistant node. Only during live
    // streaming — replay (targetEl set) is rendering an already-saved
    // record so we'd duplicate it.
    if (!targetEl && !this._streamUnifiedToolCards.some(c => c.id === ev.id)) {
      this._streamUnifiedToolCards.push({
        id: ev.id,
        tool: ev.tool,
        label: ev.label || '',
        context: ev.context || '',
      });
    }
    // Remove bulk "Using..." pill from legacy tool_status path — cards supersede it.
    const bulkRunning = container.querySelector('.pt-tool-indicator.running');
    if (bulkRunning) bulkRunning.remove();

    if (container.querySelector(`.tool-card[data-tool-id="${CSS.escape(ev.id)}"]`)) return;
    const color = _toolColor(ev.tool);
    const icon = _toolIcon(ev.tool);
    const label = ev.label || (_PT_TOOL_LABELS[ev.tool] || ev.tool.replace(/_/g, ' '));
    const category = _toolCategoryLabel(ev.tool);
    const ctx = ev.context ? `<div class="tool-card-context" title="${escapeHtml(ev.context)}">${escapeHtml(ev.context)}</div>` : '';

    const el = document.createElement('div');
    el.className = 'tool-card tool-card--live tool-card--running';
    el.dataset.toolId = ev.id;
    el.dataset.tool = ev.tool;
    el.style.setProperty('--tool-color', color);
    el.innerHTML = `
      <span class="tool-card-icon">${icon}</span>
      <div class="tool-card-body">
        <div class="tool-card-meta-row">
          <span class="tool-card-badge">${escapeHtml(category)}</span>
          <span class="tool-card-state tool-card-state--running">Working</span>
        </div>
        <div class="tool-card-head">
          <span class="tool-card-label" title="${escapeHtml(label)}">${escapeHtml(label)}</span>
        </div>
        ${ctx}
        <div class="tool-card-progress is-indeterminate"><div class="tool-card-progress-fill"></div></div>
      </div>
    `;
    container.appendChild(el);
    // Visually merge consecutive cards of the SAME tool into one continuous
    // block so a model running many searches reads as a grouped stack rather
    // than N separate floating boxes. Pure class flags — no DOM restructuring,
    // so the by-id card lookups in progress/complete are unaffected.
    const prevCard = el.previousElementSibling;
    if (prevCard && prevCard.classList && prevCard.classList.contains('tool-card--live')
        && prevCard.dataset && prevCard.dataset.tool === el.dataset.tool) {
      prevCard.classList.add('tool-card--has-next-same');
      el.classList.add('tool-card--same-as-prev');
    }
    this.updateStreamingStatus(label);
  }

  /**
   * Update a running tool card with progress or a status message.
   * @param {object} ev - { id, percent?, message? }
   */
  renderToolProgress(ev) {
    if (!ev || !ev.id) return;
    const container = this._getToolContainer();
    if (!container) return;
    const card = container.querySelector(`.tool-card[data-tool-id="${CSS.escape(ev.id)}"]`);
    if (!card) return;
    if (typeof ev.percent === 'number') {
      const bar = card.querySelector('.tool-card-progress');
      const fill = card.querySelector('.tool-card-progress-fill');
      if (bar) bar.classList.remove('is-indeterminate');
      if (fill) fill.style.width = `${Math.max(0, Math.min(100, ev.percent))}%`;
    }
    if (ev.message) {
      let msg = card.querySelector('.tool-card-message');
      if (!msg) {
        msg = document.createElement('div');
        msg.className = 'tool-card-message';
        card.querySelector('.tool-card-body').appendChild(msg);
      }
      msg.textContent = ev.message;
    }
  }

  /**
   * Transition a tool card to its final success/error state with duration.
   * @param {object} ev - { id, tool, success, duration_ms, error?, output_preview? }
   */
  renderToolComplete(ev, targetEl) {
    if (!ev || !ev.id) return;
    const container = this._getToolContainer(targetEl);
    if (!container) return;
    const card = container.querySelector(`.tool-card[data-tool-id="${CSS.escape(ev.id)}"]`);
    if (!card) return;
    // Capture completion state on the matching record so the persisted
    // node has enough to fully replay this card on next render.
    if (!targetEl) {
      const rec = this._streamUnifiedToolCards.find(c => c.id === ev.id);
      if (rec) {
        rec.success = ev.success !== false;
        if (ev.duration_ms != null) rec.duration_ms = ev.duration_ms;
        if (ev.error) rec.error = ev.error;
        if (ev.output_preview) rec.output_preview = ev.output_preview;
        if (ev.result_metadata) rec.result_metadata = ev.result_metadata;
      }
    }
    const cls = ev.success ? 'tool-card--success' : 'tool-card--error';
    card.classList.remove('tool-card--running');
    card.classList.add(cls);
    const stateEl = card.querySelector('.tool-card-state');
    if (stateEl) {
      stateEl.textContent = ev.success ? 'Ready' : 'Failed';
      stateEl.classList.remove('tool-card-state--running');
      stateEl.classList.add(ev.success ? 'tool-card-state--success' : 'tool-card-state--error');
    }
    // Swap icon to check/error
    const iconEl = card.querySelector('.tool-card-icon');
    if (iconEl) iconEl.innerHTML = ev.success ? _TOOL_ICONS_SVG.check : _TOOL_ICONS_SVG.error;
    // Add duration pill to head row
    const head = card.querySelector('.tool-card-head');
    if (head && ev.duration_ms) {
      let dur = head.querySelector('.tool-card-duration');
      if (!dur) {
        dur = document.createElement('span');
        dur.className = 'tool-card-duration';
        head.appendChild(dur);
      }
      dur.textContent = _formatDuration(ev.duration_ms);
    }
    // Flip indeterminate bar to filled/empty
    const bar = card.querySelector('.tool-card-progress');
    if (bar) bar.classList.remove('is-indeterminate');
    const body = card.querySelector('.tool-card-body');
    // Rendered on failure too: code that was rejected or that errored is
    // exactly when the user most needs to see what was actually run. The
    // per-tool renderers return '' when their metadata is absent, so a
    // genuinely empty failure still falls through to the error line below.
    const resultViewHtml = renderToolResultView(ev.tool, ev.result_metadata || {});
    if (body && resultViewHtml && !body.querySelector('.tool-result-section')) {
      const wrap = document.createElement('div');
      wrap.className = 'tool-card-result-view';
      wrap.innerHTML = resultViewHtml;
      body.appendChild(wrap);
    }
    // Just-in-time push subscription prompt for schedule_briefing.
    // The substrate fires Web Push when the user's tab is closed,
    // but only if they've subscribed. After a successful schedule
    // is the natural moment to ask — the value is concrete and
    // immediate. Module no-ops if already subscribed or dismissed.
    if (ev.success && ev.tool === 'schedule_briefing' && body
        && !body.querySelector('.push-prompt-card')) {
      import('../notifications/push-prompt.js').then((mod) => {
        mod.mountPushPromptIfNeeded(body, {
          headline: 'Want this briefing to reach you when the tab is closed?',
          body: 'Enable browser notifications so the daily briefing buzzes your device even when Augmentum isn’t open.',
        }).catch((e) => { console.warn('[push-prompt] mount failed', e); });
      }).catch((e) => { console.warn('[push-prompt] load failed', e); });
    }
    if (ev.success && ev.output_preview && body && !resultViewHtml) {
      let msg = card.querySelector('.tool-card-message');
      if (!msg) {
        msg = document.createElement('div');
        msg.className = 'tool-card-message';
        body.appendChild(msg);
      }
      msg.textContent = ev.output_preview;
      msg.title = ev.output_preview;
    }
    // Surface error details if present and the tool failed
    if (!ev.success && ev.error) {
      if (body && !body.querySelector('.tool-card-error')) {
        const err = document.createElement('div');
        err.className = 'tool-card-error';
        err.textContent = ev.error;
        err.title = ev.error;
        body.appendChild(err);
      }
    }
    // Tool-specific result rendering — show actual images returned by
    // image_search directly under the card. Without this the model has
    // to describe URLs in prose, which it does poorly.
    if (ev.success && ev.tool === 'image_search') {
      const rm = ev.result_metadata || ev;
      const urls = Array.isArray(rm.embed_urls) && rm.embed_urls.length
        ? rm.embed_urls
        : (Array.isArray(rm.images)
            ? rm.images.map(i => i.embed_url || i.download_url || i.source_url).filter(Boolean)
            : []);
      if (urls.length) {
        if (body && !body.querySelector('.tool-card-images')) {
          const gallery = document.createElement('div');
          gallery.className = 'tool-card-images';
          gallery.innerHTML = urls.slice(0, 6).map((u, i) => {
            const meta = Array.isArray(rm.images) ? rm.images[i] : null;
            const title = meta?.title || '';
            const source = meta?.source || '';
            const caption = title || source ? `
              <span class="tool-card-image-caption">
                ${title ? `<span class="tool-card-image-title">${escapeHtml(title)}</span>` : ''}
                ${source ? `<span class="tool-card-image-source">${escapeHtml(source)}</span>` : ''}
              </span>
            ` : '';
            return `<a class="tool-card-image" href="${escapeHtml(meta?.source_url || u)}" target="_blank" rel="noopener" title="${escapeHtml(title + (source ? ' — ' + source : ''))}">
              <img src="${escapeHtml(u)}" alt="${escapeHtml(title)}" loading="lazy" decoding="async" onerror="this.parentElement.remove()">
              ${caption}
            </a>`;
          }).join('');
          body.appendChild(gallery);
        }
      }
    }
  }

  /**
   * Show a chain plan indicator on the streaming message.
   * @param {object} chain - Chain state { total_steps, source }
   */
  showChainPlanIndicator(chain) {
    const container = this._getToolContainer();
    if (!container) return;
    if (container.querySelector('.chain-plan-label')) return;
    const total = chain.total_steps || '?';
    const source = chain.source?.startsWith('custom:') ? 'flow' : 'chain';
    const el = document.createElement('div');
    el.className = 'chain-plan-label';
    el.textContent = `${source === 'flow' ? 'Running flow' : 'Multi-step'} \u00b7 ${total} steps`;
    container.prepend(el);
  }

  /**
   * Finalize the streaming message — remove cursor, add action buttons, return element.
   * @param {object}  session  - Current session
   * @param {boolean} complete - Whether the stream completed normally
   * @returns {HTMLElement|null} The finalized element
   */
  finalizeStreaming(session, complete = true) {
    const el = this.streamingEl;
    if (!el) return null;

    let rawContent = el.dataset.rawContent || '';
    el.removeAttribute('data-raw-content');

    // Remove streaming cursor, re-render final content. If the rAF flush
    // scheduled by appendToStreaming never fired before the stream ended
    // (short response, all-at-once buffered delivery, or background tab
    // where rAF is throttled), .response-body won't exist yet and the
    // streaming-dots placeholder is still in the DOM. Build the body and
    // drop the dots so the bubble doesn't sit on "Processing prompt…"
    // forever — the assistant node is already being saved with rawContent,
    // so without this, refresh fills in what the live UI never did.
    let responseBody = el.querySelector('.response-body');
    if (!responseBody) {
      const messageContent = el.querySelector('.message-content');
      if (messageContent) {
        const dots = messageContent.querySelector('.streaming-dots');
        if (dots) dots.remove();
        responseBody = document.createElement('div');
        responseBody.className = 'response-body';
        messageContent.appendChild(responseBody);
      }
    }
    if (responseBody) {
      responseBody.classList.remove('streaming-cursor');
      // Final render via the SAME bounded incremental renderer the live
      // stream uses — NOT a from-scratch ``renderMarkdown(rawContent)``.
      // The from-scratch path fed the ENTIRE (possibly huge) message to the
      // regex pipeline in one synchronous call, then highlighted
      // synchronously — blocking the main thread for seconds-to-minutes on a
      // long generation (the bold/italic regexes can backtrack
      // catastrophically on delimiter-heavy text). renderStreamSplit promotes
      // only the still-unsettled tail, in safe bounded chunks, and defers
      // highlight to idle — so finalize is O(tail), never O(whole message),
      // and never freezes the page after the gen completes.
      const messageContent = responseBody.parentElement
        || el.querySelector('.message-content');
      if (!this._streamSplit) this._streamSplit = newSplitState();
      if (messageContent) {
        renderStreamSplit(messageContent, rawContent, this._streamSplit, {
          mode: this._mode,
          narrativePanelsCollapsed: this._narrativePanelsCollapsed,
          highlightHooks: this._highlightHooks,
        });
      } else {
        // No container to host the split scaffold (shouldn't happen) — fall
        // back to a single render, but still defer the highlight below.
        responseBody.innerHTML = renderMarkdown(rawContent, { mode: this._mode, narrativePanelsCollapsed: this._narrativePanelsCollapsed });
      }
      // Run the post-processing hooks (mermaid / artifact cards / code
      // versions / KaTeX) + any remaining highlight on idle, over the whole
      // body. highlightCode self-skips already-highlighted blocks, so this is
      // cheap + idempotent and never blocks the finalize frame.
      highlightCodeDeferred(responseBody, this._highlightHooks);
    }

    // Collapse thinking blocks + reasoning summaries
    el.querySelectorAll('.thinking-block.open').forEach(b => b.classList.remove('open'));
    el.querySelectorAll('.reasoning-summary.open').forEach(b => b.classList.remove('open'));
    const modelBlock = el.querySelector('.model-thinking-block');
    if (modelBlock) {
      const label = modelBlock.querySelector('.thinking-header span');
      if (label) label.textContent = 'Thought for a moment';
    }

    // Store raw content for delegated copy/TTS handlers
    el.dataset.rawContent = rawContent;

    // Add action buttons (delegated — no per-element listeners)
    const nodeId = el.dataset.nodeId || 'pending';
    const bubble = el.querySelector('.message-bubble');
    if (bubble) {
      // Structural interrupted badge — sits between body and chrome.
      // Read from session node (host wrote node.interrupted in onError
      // before calling finalizeStreaming with complete=false). Falls
      // back to the complete arg when no node is wired yet.
      const node = session && nodeId !== 'pending' ? session.tree?.[nodeId] : null;
      const isInterrupted = node ? !!node.interrupted : !complete;
      if (isInterrupted) {
        const badge = document.createElement('div');
        badge.className = 'interrupted-badge';
        badge.setAttribute('role', 'status');
        const msg = (node && node.error_message) || '';
        if (msg) badge.title = msg;
        badge.textContent = 'Response interrupted';
        bubble.appendChild(badge);
      }

      // Branch swipe indicator — without this, a freshly regenerated reply
      // has no siblings nav until the next full renderMessages (typically a
      // page refresh). Only appears when the node has ≥2 same-role siblings,
      // so the normal first-gen case is unaffected.
      if (session && nodeId !== 'pending') {
        const siblingInfo = getSiblingInfo(session, nodeId);
        if (siblingInfo) {
          const swipe = document.createElement('div');
          swipe.className = 'swipe-controls';
          swipe.innerHTML = `
            <button class="swipe-btn" data-action="branch-prev" data-node-id="${nodeId}" ${siblingInfo.index === 0 ? 'disabled' : ''}>${icons.chevronLeft}</button>
            <span class="swipe-indicator">${siblingInfo.index + 1}/${siblingInfo.total}</span>
            <button class="swipe-btn" data-action="branch-next" data-node-id="${nodeId}" ${siblingInfo.index === siblingInfo.total - 1 ? 'disabled' : ''}>${icons.chevronRight}</button>
          `;
          bubble.appendChild(swipe);
        }
      }

      // Use the same shape as createMessageEl so the row width doesn't
      // change between fresh-finalize and the next renderMessages rebuild.
      // The just-finalized message IS the last assistant by definition.
      const actionsDiv = document.createElement('div');
      actionsDiv.className = 'message-actions';
      actionsDiv.innerHTML = this._buildAssistantActionsInnerHtml(nodeId, {
        isLastAssistant: true,
        isNarrativeSession: session?.mode === 'narrative',
      });
      bubble.appendChild(actionsDiv);

      // Knowledge-pack chip \u2014 read from the assistant node which the host
      // populated just before calling finalizeStreaming. Without this the
      // chip only appears after a full renderMessages() rebuild (page
      // refresh, branch nav, next user turn), making the just-streamed
      // turn feel like the feature isn't working.
      if (node && node.knowledgePack) {
        const chipHtml = _renderKnowledgePackChip(node.knowledgePack);
        if (chipHtml) {
          const chipWrapper = document.createElement('div');
          chipWrapper.innerHTML = chipHtml;
          const chipEl = chipWrapper.firstElementChild;
          if (chipEl) {
            // Match the renderMessage template order: chip \u2192 speed \u2192 actions
            bubble.insertBefore(chipEl, actionsDiv);
          }
        }
      }

      // Generation stats
      const m = this._streamMetrics;
      const statParts = _buildGenerationStatParts(m);
      if (statParts.length > 0) {
        const speedEl = document.createElement('div');
        speedEl.className = 'message-gen-speed';
        speedEl.title = _generationStatsTitle(m);
        speedEl.textContent = statParts.join(' \u00B7 ');
        // Insert before actions
        bubble.insertBefore(speedEl, actionsDiv);
      }
    }

    const finalized = this.streamingEl;
    this.streamingEl = null;
    // Mirror of the start-signal in appendToStreaming. Covers normal
    // completion and abort (complete=false), since both land here.
    if (this._streamSignaled) {
      this._streamSignaled = false;
      bus.set('chat_streaming', false);
    }
    this.scrollToBottom();
    return finalized;
  }

  /**
   * Get the current raw content from the streaming element.
   * @returns {string}
   */
  getStreamingRawContent() {
    return this.streamingEl?.dataset.rawContent || '';
  }

  /**
   * Get the current streaming element's node ID.
   * @returns {string}
   */
  getStreamingNodeId() {
    return this.streamingEl?.dataset.nodeId || '';
  }

  /**
   * Update the node ID on the streaming element (called after the node is committed to the tree).
   * @param {string} nodeId
   */
  setStreamingNodeId(nodeId) {
    if (this.streamingEl) this.streamingEl.dataset.nodeId = nodeId;
  }

  // -----------------------------------------------------------------------
  // Stream state
  // -----------------------------------------------------------------------

  /** Reset all stream state properties to initial values. */
  resetStreamState() {
    this._streamPhases       = [];
    this._streamToolCalls    = [];
    this._streamUnifiedToolCards = [];
    this._streamNarrativeActivity = [];
    this._streamThinking     = '';
    this._streamThinkingOpen = false;
    this._streamPhaseContent = {};
    this._streamComplexity   = '';
    this._streamFlowName     = '';
    this._streamSpeaker      = '';
    // Match the constructor field set exactly — drift here meant
    // promptTokensEvaluated/Cached leaked across turns until the next
    // metrics tick overwrote them, briefly showing last-turn's cached
    // count in the ambient stats bar.
    this._streamMetrics      = {
      tps: 0,
      contextLen: 0,
      contextUsed: 0,
      promptTokens: 0,
      promptTokensEvaluated: 0,
      promptTokensCached: 0,
      promptTokensCacheWrite: 0,
      kvReuse: '',
      kvVoidCause: '',
      evalTokens: 0,
      ttftMs: 0,
      totalDurationMs: 0,
      evalDurationMs: 0,
      promptTokensEstimated: false,
      evalTokensEstimated: false,
    };
    this._streamRenderScheduled = false;
    this._streamThinkRenderScheduled = false;
    this._streamSplit = newSplitState();
  }

  setStreamSpeaker(name) { this._streamSpeaker = name || ''; }
  getStreamSpeaker() { return this._streamSpeaker; }

  /**
   * Build the collected reasoning data for persistence on the finalized node.
   * Includes model-native `thinking` (reasoning_content / <think> tokens) so
   * refresh + server restart round-trip without losing it — important for
   * later training-data extraction and just for the user not to see a chain
   * of thought vanish after a page reload.
   * @returns {object|null}
   */
  collectReasoningData() {
    if (this._streamPhases.length === 0 && !this._streamThinking) return null;
    const data = {
      phases:       this._streamPhases,
      toolCalls:    this._streamToolCalls,
      phaseContent: this._streamPhaseContent,
      complexity:   this._streamComplexity,
      flow_name:    this._streamFlowName,
    };
    if (this._streamThinking) data.thinking = this._streamThinking;
    return data;
  }

  /**
   * Snapshot the unified tool-card records captured during this stream.
   * Caller (index.js finalizeStreaming) writes this onto the assistant
   * node so replayToolCards() can reconstruct the cards on later
   * renderMessages() passes.
   */
  collectToolCards() {
    return this._streamUnifiedToolCards.slice();
  }

  /**
   * Re-render persisted tool cards into a finalized message element.
   * Drives the same renderToolStart/renderToolComplete code path as the
   * live stream — so the result-section gallery (image_search etc.)
   * gets reconstructed from result_metadata without duplicating render
   * logic here.
   */
  replayToolCards(toolCards, messageEl) {
    if (!Array.isArray(toolCards) || !messageEl) return;
    for (const tc of toolCards) {
      if (!tc || !tc.id) continue;
      this.renderToolStart({
        id: tc.id,
        tool: tc.tool,
        label: tc.label,
        context: tc.context,
      }, messageEl);
      if (tc.success != null) {
        this.renderToolComplete({
          id: tc.id,
          tool: tc.tool,
          success: tc.success,
          duration_ms: tc.duration_ms,
          error: tc.error,
          output_preview: tc.output_preview,
          result_metadata: tc.result_metadata,
        }, messageEl);
      }
    }
  }

  // -----------------------------------------------------------------------
  // Narrative internal-tool activity trail
  // -----------------------------------------------------------------------

  /**
   * Record one narrative internal-tool action and render it live into the
   * streaming message's activity strip. ``entry`` = {tool, args,
   * resultPreview, kind}. ``kind`` distinguishes 'tool' (a real call),
   * 'suppressed' (a dropped tool-plan preamble) and 'synth' (the forced
   * tool-free synthesis pass) so the strip can style them differently.
   */
  addNarrativeActivity(entry) {
    if (!entry) return;
    const rec = {
      tool: entry.tool || '',
      args: entry.args || null,
      resultPreview: entry.resultPreview || '',
      kind: entry.kind || 'tool',
      label: entry.label || _narrativeActivityLabel(entry.tool, entry.args),
    };
    this._streamNarrativeActivity.push(rec);
    // Also reflect the current action in the streaming status line so the
    // silent loader is replaced with what's actually happening.
    if (rec.kind === 'tool') this.updateStreamingStatus(rec.label);

    const el = this.streamingEl;
    if (!el) return;
    const content = el.querySelector('.message-content') || el;
    let strip = content.querySelector('.narrative-activity');
    if (!strip) {
      strip = document.createElement('div');
      strip.className = 'narrative-activity narrative-activity--live';
      strip.innerHTML =
        '<div class="narrative-activity-head">'
        + '<span class="narrative-activity-label">Working</span>'
        + '<span class="narrative-activity-count"></span></div>'
        + '<ol class="narrative-activity-log" aria-live="polite"></ol>';
      // Sit above the streaming dots / content so it reads as "what led
      // to this reply", not part of the prose.
      const dots = content.querySelector('.streaming-dots');
      if (dots) content.insertBefore(strip, dots);
      else content.insertBefore(strip, content.firstChild);
    }
    const log = strip.querySelector('.narrative-activity-log');
    if (log) {
      const li = document.createElement('li');
      li.className = `narrative-activity-item narrative-activity-item--${rec.kind}`;
      const detail = (rec.resultPreview || '').trim();
      if (detail) {
        li.classList.add('narrative-activity-item--expandable');
        const d = document.createElement('details');
        const s = document.createElement('summary');
        s.textContent = rec.label;
        const body = document.createElement('div');
        body.className = 'narrative-activity-detail';
        body.textContent = detail;
        d.appendChild(s);
        d.appendChild(body);
        li.appendChild(d);
      } else {
        li.textContent = rec.label;
      }
      log.appendChild(li);
    }
    const toolCount = this._streamNarrativeActivity.filter(a => a.kind === 'tool').length;
    const countEl = strip.querySelector('.narrative-activity-count');
    if (countEl) countEl.textContent = toolCount ? `${toolCount}` : '';
  }

  /**
   * Collapse the live activity strip into its done chip mid-stream (called
   * on the first prose token). Swaps the expanded ``--live`` strip for the
   * same collapsed ``<details>`` that finalize/replay produce, so the
   * transition is seamless. No-op if there's no live strip.
   */
  collapseNarrativeActivity() {
    const el = this.streamingEl;
    if (!el) return;
    const content = el.querySelector('.message-content') || el;
    const live = content.querySelector('.narrative-activity--live');
    if (!live) return;
    live.remove();
    if (this._streamNarrativeActivity.length) {
      this.replayNarrativeActivity(this._streamNarrativeActivity, el);
    }
  }

  /**
   * Snapshot the narrative activity captured during this stream. Caller
   * (index.js finalizeStreaming) writes this onto the assistant node so
   * replayNarrativeActivity() can rebuild the collapsed chip on later
   * renderMessages() passes.
   */
  collectNarrativeActivity() {
    return this._streamNarrativeActivity.slice();
  }

  /**
   * Render persisted narrative activity into a finalized message as a
   * collapsed, expandable chip ("Consulted memory · N lookups"). Story-
   * first: closed by default so it never competes with the prose.
   */
  replayNarrativeActivity(items, messageEl) {
    if (!Array.isArray(items) || !items.length || !messageEl) return;
    const content = messageEl.querySelector('.message-content') || messageEl;
    if (content.querySelector('.narrative-activity')) return;  // idempotent
    const tools = items.filter(a => a.kind === 'tool');
    const reads = tools.filter(a => /recall|check|search|list/.test(a.tool)).length;
    const writes = tools.length - reads;
    const parts = [];
    if (reads) parts.push(`${reads} lookup${reads === 1 ? '' : 's'}`);
    if (writes) parts.push(`${writes} lore edit${writes === 1 ? '' : 's'}`);
    const summary = parts.length ? parts.join(' · ') : 'no tools used';
    const rows = items.map(a => {
      const label = escapeHtml(a.label || _narrativeActivityLabel(a.tool, a.args));
      const detail = (a.resultPreview || '').trim();
      const cls = `narrative-activity-item narrative-activity-item--${a.kind || 'tool'}`;
      if (!detail) return `<li class="${cls}">${label}</li>`;
      return `<li class="${cls} narrative-activity-item--expandable">`
        + `<details><summary>${label}</summary>`
        + `<div class="narrative-activity-detail">${escapeHtml(detail)}</div>`
        + `</details></li>`;
    }).join('');
    const details = document.createElement('details');
    details.className = 'narrative-activity narrative-activity--done';
    details.innerHTML =
      `<summary class="narrative-activity-summary">`
      + `<span class="narrative-activity-icon" aria-hidden="true">◈</span>`
      + `<span>Consulted memory · ${escapeHtml(summary)}</span></summary>`
      + `<ol class="narrative-activity-log">${rows}</ol>`;
    content.insertBefore(details, content.firstChild);
  }

  // -----------------------------------------------------------------------
  // Scroll
  // -----------------------------------------------------------------------

  /**
   * Scroll to the bottom of the chat area.
   * @param {boolean} smooth - Use smooth scrolling (default true)
   * @param {boolean} force  - Force scroll even if user scrolled up (default false)
   */
  scrollToBottom(smooth = true, force = false) {
    if (!force && this.userScrolledUp) return;
    // CSS @media (prefers-reduced-motion) clamps animation/transition
    // durations globally, but JS scrollTo({behavior:'smooth'}) is its own
    // animation that CSS can't reach. Snap-scroll instead when reduced.
    const behavior = smooth && !_prefersReducedMotion() ? 'smooth' : 'instant';
    this.scrollEl?.scrollTo({
      top: this.scrollEl.scrollHeight,
      behavior,
    });
  }

  // -----------------------------------------------------------------------
  // Empty state
  // -----------------------------------------------------------------------

  /**
   * Show or hide the empty state based on whether the session has messages.
   * @param {boolean} hasMessages
   */
  updateEmptyState(hasMessages) {
    if (this.emptyStateEl) {
      this.emptyStateEl.style.display = hasMessages ? 'none' : '';
    }
  }

  // -----------------------------------------------------------------------
  // Mode
  // -----------------------------------------------------------------------

  /**
   * Update the rendering mode (affects markdown rendering, e.g. narrative dialogue).
   * @param {string} mode
   */
  setMode(mode) {
    this._mode = mode;
  }

  /** Set whether ```md / ```stats / ```scene blocks render collapsed by
   *  default. Sourced from the active character's autoCollapseNarrativePanels
   *  preference. Per-element user toggles still override this at runtime. */
  setNarrativePanelsCollapsed(collapsed) {
    this._narrativePanelsCollapsed = collapsed !== false;
  }

  // -----------------------------------------------------------------------
  // Highlighting
  // -----------------------------------------------------------------------

  /** @private */
  _highlightAll(container) {
    highlightCode(container, this._highlightHooks);
  }

  // -----------------------------------------------------------------------
  // Thinking-panel builders — shared between live UARF, live model-native,
  // and stored renders so the three paths stay visually + behaviorally
  // identical. Variants:
  //   - 'live'   : streaming indicator (header shows running phase,
  //                pipeline + minimal phase rows, no footer, no template)
  //   - 'stored' : post-mortem (header shows confidence badge, phase rows
  //                carry descriptions + tool sub-rows, footer with
  //                Inspector button, hidden <template> carrying the raw
  //                JSON for the Inspector)
  // -----------------------------------------------------------------------

  /** @private — model-native reasoning_content / <think> block. */
  _buildModelThinkingHtml(thinking, opts = {}) {
    if (!thinking) return '';
    const variant = opts.variant || 'stored';
    const label = variant === 'stored' ? 'Thought for a moment' : 'Thinking\u2026';
    const open = opts.open !== false && variant === 'live';
    const openCls = open ? ' open' : '';
    const expanded = open ? 'true' : 'false';
    // <button> instead of <div> so keyboard users get focus + Enter/Space
    // toggle for free, and AT reads it as a button. CSS resets are in
    // chat.css under .thinking-header so it doesn't look like a default
    // submit button. Click handling is delegated in _initDelegatedActions.
    return `<div class="thinking-block model-thinking-block${openCls}">
      <button type="button" class="thinking-header" aria-expanded="${expanded}">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><polyline points="6 9 12 15 18 9"/></svg>
        <span>${label}</span>
      </button>
      <div class="thinking-content"><pre class="model-thinking-pre" style="white-space:pre-wrap;font-size:var(--text-sm);color:var(--text-muted);margin:0">${escapeHtml(thinking)}</pre></div>
    </div>`;
  }

  /** @private — UARF phases pipeline + per-phase rows. ``data`` carries
   *  ``{phases, toolCalls, phaseContent, flowName, confidence}``; the
   *  full ``rawReasoning`` only matters for the stored variant which
   *  embeds it for the Inspector button. */
  _buildPhasesPanelHtml(data = {}, opts = {}) {
    const phases       = data.phases || [];
    if (phases.length === 0) return '';
    const variant      = opts.variant || 'stored';
    const summaryOpen  = !!opts.summaryOpen;
    const toolCalls    = data.toolCalls || [];
    const phaseContent = data.phaseContent || {};
    const flowName     = data.flowName || data.flow_name || '';
    const confidence   = data.confidence;
    const rawReasoning = opts.rawReasoning || null;

    const completedCount = phases.filter(p => p.status === 'complete').length;
    const totalCount     = phases.length;
    const runningPhase   = phases.find(p => p.status === 'running');
    const toolCount      = toolCalls.length;

    // Header label — live focuses on progress, stored shows totals.
    const headerParts = [];
    if (flowName) headerParts.push(escapeHtml(flowName));
    if (variant === 'live') {
      if (runningPhase) {
        headerParts.push(escapeHtml(PHASE_DISPLAY_NAMES[runningPhase.name] || runningPhase.name) + '\u2026');
      } else {
        headerParts.push(`${completedCount}/${totalCount} phases`);
      }
    } else {
      headerParts.push(`${totalCount} phase${totalCount !== 1 ? 's' : ''}`);
      if (toolCount > 0) headerParts.push(`${toolCount} tool${toolCount !== 1 ? 's' : ''}`);
    }
    const headerLabel = headerParts.join(' \u00B7 ');

    // Metrics slot — live shows tool count, stored shows confidence badge.
    let metricsHtml = '';
    if (variant === 'live' && toolCount > 0) {
      metricsHtml = `<span class="reasoning-summary__metric">${toolCount} tool${toolCount !== 1 ? 's' : ''}</span>`;
    } else if (variant === 'stored' && confidence != null && typeof confidence === 'number') {
      const pct = Math.round(confidence > 1 ? confidence : confidence * 100);
      const level = pct >= 80 ? 'high' : pct >= 50 ? 'medium' : 'low';
      const label = pct >= 80 ? 'High' : pct >= 50 ? 'Medium' : 'Low';
      metricsHtml = `<span class="reasoning-summary__confidence reasoning-summary__confidence--${level}">${label}</span>`;
    }

    const pipelineDots = phases.map((p, i) => {
      const status = _safeStatus(p.status);
      const conn = i > 0 ? '<span class="reasoning-summary__pipe-conn"></span>' : '';
      const title = ` title="${escapeHtml(PHASE_DISPLAY_NAMES[p.name] || p.name)}"`;
      return `${conn}<span class="reasoning-summary__pipe-dot reasoning-summary__pipe-dot--${status}"${title}></span>`;
    }).join('');

    const phaseRows = phases.map(p => {
      const displayName = PHASE_DISPLAY_NAMES[p.name] || p.name;
      const statusIcon = p.status === 'complete' ? '\u2713'
        : p.status === 'running' ? '\u25CF'
        : '\u25CB';
      const phaseTools = toolCalls.filter(tc => tc.phase === p.name);
      const toolCountLabel = phaseTools.length > 0
        ? `\u2014 ${phaseTools.length} tool${phaseTools.length !== 1 ? 's' : ''}`
        : '';

      let descHtml = '';
      let toolRows = '';
      if (variant === 'stored') {
        const content = phaseContent[p.name] || '';
        const desc = content.split('\n').filter(l => l.trim()).slice(0, 1).join('').substring(0, 120);
        if (desc) descHtml = `<span class="reasoning-summary__phase-desc">${escapeHtml(desc)}</span>`;
        toolRows = phaseTools.map(tc => {
          const statusCls = tc.success === true ? 'success' : tc.success === false ? 'error' : 'success';
          const statusChar = tc.success === false ? '\u2717' : '\u2713';
          const inputStr = typeof tc.input === 'string' ? tc.input
            : tc.input ? JSON.stringify(tc.input) : '';
          const inputPreview = inputStr.length > 80 ? inputStr.substring(0, 80) + '\u2026' : inputStr;
          return `<div class="reasoning-summary__tool">
            <span class="reasoning-summary__tool-status reasoning-summary__tool-status--${statusCls}">${statusChar}</span>
            <span class="reasoning-summary__tool-name">${escapeHtml(tc.tool || '')}</span>
            ${inputPreview ? `<span>(${escapeHtml(inputPreview)})</span>` : ''}
          </div>`;
        }).join('');
      }

      return `<div class="reasoning-summary__phase">
        <span class="reasoning-summary__phase-icon">${statusIcon}</span>
        <span class="reasoning-summary__phase-name">${escapeHtml(displayName)}</span>
        <span class="reasoning-summary__phase-tools">${toolCountLabel}</span>
        ${descHtml}
      </div>${toolRows}`;
    }).join('');

    let footerHtml = '';
    let templateHtml = '';
    if (variant === 'stored') {
      // No inline onclick; .reasoning-summary__inspect clicks are caught
      // by the delegated handler in _initDelegatedActions.
      footerHtml = `<div class="reasoning-summary__footer">
        <button type="button" class="reasoning-summary__inspect">Open in Inspector \u2192</button>
      </div>`;
      if (rawReasoning) {
        templateHtml = `<template class="stored-reasoning-data">${escapeHtml(JSON.stringify(rawReasoning))}</template>`;
      }
    }

    const openCls = summaryOpen ? ' open' : '';
    const expanded = summaryOpen ? 'true' : 'false';
    const dataHas = variant === 'stored' ? ' data-has-reasoning="true"' : '';

    return `<div class="reasoning-summary${openCls}"${dataHas}>
      <button type="button" class="reasoning-summary__header" aria-expanded="${expanded}">
        <span class="reasoning-summary__diamond" aria-hidden="true"></span>
        <span class="reasoning-summary__label">${headerLabel}</span>
        <span class="reasoning-summary__metrics">
          ${metricsHtml}
          <svg class="reasoning-summary__toggle" viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 6 8 10 12 6"/></svg>
        </span>
      </button>
      <div class="reasoning-summary__body">
        <div class="reasoning-summary__pipeline" aria-hidden="true">${pipelineDots}</div>
        ${phaseRows}
        ${footerHtml}
      </div>
    </div>${templateHtml}`;
  }

  /** @private — stored reasoning panel: model-thinking + phases combined. */
  _buildStoredThinkingHtml(reasoning) {
    const modelHtml = this._buildModelThinkingHtml(reasoning.thinking, {
      variant: 'stored',
      open: false,
    });
    const phasesHtml = this._buildPhasesPanelHtml({
      phases: reasoning.phases,
      toolCalls: reasoning.toolCalls,
      phaseContent: reasoning.phaseContent,
      flowName: reasoning.flow_name,
      confidence: reasoning.confidence,
    }, {
      variant: 'stored',
      summaryOpen: false,
      rawReasoning: reasoning,
    });
    return `${modelHtml}${phasesHtml}`;
  }

  // -----------------------------------------------------------------------
  // Cleanup
  // -----------------------------------------------------------------------

  /** Remove all DOM and listeners. */
  destroy() {
    // Release any in-flight chat-streaming signal so other subsystems
    // (avatar snapshot timer, future consumers) don't see a stuck
    // "stream active" state after the surface is gone.
    if (this._streamSignaled) {
      this._streamSignaled = false;
      bus.set('chat_streaming', false);
    }
    this._closeActionOverflow();
    this._releaseScrollAnchor();
    if (this._boundScrollHandler && this.scrollEl) {
      this.scrollEl.removeEventListener('scroll', this._boundScrollHandler);
    }
    if (this._boundClickHandler && this.messagesEl) {
      this.messagesEl.removeEventListener('click', this._boundClickHandler);
    }
    if (this._boundViewportResize && window.visualViewport) {
      window.visualViewport.removeEventListener(
        'resize', this._boundViewportResize,
      );
    }
    if (this._boundWindowResize && typeof window !== 'undefined') {
      window.removeEventListener('resize', this._boundWindowResize);
    }
    if (this.scrollEl && this.scrollEl.parentNode) {
      this.scrollEl.parentNode.removeChild(this.scrollEl);
    }
    this.messagesEl        = null;
    this.scrollEl          = null;
    this.emptyStateEl      = null;
    this.streamingEl       = null;
    this._boundScrollHandler = null;
    this._boundClickHandler  = null;
    this._boundViewportResize = null;
    this._boundWindowResize  = null;
  }
}
