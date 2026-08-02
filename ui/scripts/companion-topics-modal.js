/**
 * companion-topics-modal.js — the Schedule surface.
 *
 * A professional manager for the companion's standing tasks: briefings,
 * scheduled requests & actions, watches, and feed/search digests, plus
 * the lightweight "topics" watch-list. Opened from the header menu, the
 * command palette ("Open Schedule"), and the notes drawer gear.
 *
 * Design:
 *   - Grouped by purpose (Briefings · Requests & actions · Watches ·
 *     Topics), not by raw kind.
 *   - A purpose-first "New" flow: pick what you want, get a form tailored
 *     to that kind (schema-driven), with real time-of-day / weekday /
 *     one-shot scheduling.
 *   - Each task shows a human schedule, how it's delivered (alerts every
 *     device vs. quiet digest), its status, and an expandable run history
 *     (the trust surface: "checked 2h ago, nothing new").
 *   - Real edit (PATCH title/params/schedule) and run-now / pause / delete.
 *
 * Persona-agnostic copy — this is shipped chrome (see the OSS labels rule);
 * spoken/prompted persona lives elsewhere. The backend contract is owned
 * by augmentum/proxy/companion_routes.py + companion_runtime/standing_tasks.py.
 */

import { installDialog } from './_focus-trap.js';

let _modal = null;
let _dialog = null;       // focus-trap handle, live only while shown
let _kinds = [];          // kinds the backend reports it can run
let _view = 'list';       // 'list' | 'form'
let _editing = null;      // task being edited, or null for create

function _esc(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/`/g, '&#96;')
    .replace(/\$\{/g, '&#36;{');
}

// ── API ─────────────────────────────────────────────────────────────

async function _fetchTopics() {
  try {
    const resp = await fetch('/api/companion/topics', { credentials: 'same-origin' });
    if (!resp.ok) return [];
    const data = await resp.json();
    return Array.isArray(data.topics) ? data.topics : [];
  } catch (_) { return []; }
}

async function _addTopic(topic) {
  try {
    const resp = await fetch('/api/companion/topics', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ topic }),
    });
    if (resp.status === 409) return { ok: false, reason: 'duplicate' };
    if (!resp.ok) return { ok: false, reason: 'server' };
    return await resp.json();
  } catch (_) { return { ok: false, reason: 'network' }; }
}

async function _removeTopic(id) {
  try {
    const resp = await fetch(`/api/companion/topics/${id}`, {
      method: 'DELETE', credentials: 'same-origin',
    });
    return resp.ok;
  } catch (_) { return false; }
}

async function _fetchTasks() {
  try {
    const resp = await fetch('/api/companion/tasks', { credentials: 'same-origin' });
    if (!resp.ok) return { tasks: [], kinds: [] };
    const data = await resp.json();
    return {
      tasks: Array.isArray(data.tasks) ? data.tasks : [],
      kinds: Array.isArray(data.kinds) ? data.kinds : [],
    };
  } catch (_) { return { tasks: [], kinds: [] }; }
}

async function _addTask(payload) {
  try {
    const resp = await fetch('/api/companion/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      let reason = 'server';
      try { reason = (await resp.json()).reason || reason; } catch (_) {}
      return { ok: false, reason };
    }
    return await resp.json();
  } catch (_) { return { ok: false, reason: 'network' }; }
}

async function _cronPreview(expr) {
  // Live schedule-builder feedback: server-side validation + human
  // gloss + next 3 fire times in the user's timezone. One rule for
  // "what counts as valid cron" — the server's — so the builder can
  // never disagree with the engine.
  try {
    const resp = await fetch('/api/companion/tasks/cron-preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ cron: expr }),
    });
    if (!resp.ok) return { ok: false, error: 'preview unavailable' };
    return await resp.json();
  } catch (_) { return { ok: false, error: 'network' }; }
}

async function _resolveFeed(source) {
  // Server-side source resolution: YouTube channel/@handle → keyless
  // Atom feed, r/name → subreddit RSS, site URL → autodiscovery. The
  // response carries the feed title + latest entry so the save is a
  // confirmed, working follow — never a dead watch.
  try {
    const resp = await fetch('/api/companion/tasks/resolve-feed', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ source }),
    });
    if (!resp.ok) return { ok: false, error: 'resolver unavailable' };
    return await resp.json();
  } catch (_) { return { ok: false, error: 'network' }; }
}

async function _patchTask(id, payload) {
  try {
    const resp = await fetch(`/api/companion/tasks/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      let reason = 'server';
      try { reason = (await resp.json()).reason || reason; } catch (_) {}
      return { ok: false, reason };
    }
    return await resp.json();
  } catch (_) { return { ok: false, reason: 'network' }; }
}

async function _removeTask(id) {
  try {
    const resp = await fetch(`/api/companion/tasks/${id}`, {
      method: 'DELETE', credentials: 'same-origin',
    });
    return resp.ok;
  } catch (_) { return false; }
}

async function _runNow(id) {
  try {
    const resp = await fetch(`/api/companion/tasks/${id}/run-now`, {
      method: 'POST', credentials: 'same-origin',
    });
    if (!resp.ok) return null;
    return await resp.json();
  } catch (_) { return null; }
}

async function _fetchRuns(id) {
  try {
    const resp = await fetch(`/api/companion/tasks/${id}/runs`, { credentials: 'same-origin' });
    if (!resp.ok) return [];
    const data = await resp.json();
    return Array.isArray(data.runs) ? data.runs : [];
  } catch (_) { return []; }
}

// ── Purpose schema ──────────────────────────────────────────────────
//
// Each purpose maps to one backend kind and declares the fields its
// params need. The renderer + assembler are generic over field `type`.
// `group` buckets it in the New picker and the list. `delivery` mirrors
// the backend importance routing (active-delivery kinds alert every
// device; the rest stay quiet until you're away).

const GROUPS = [
  { id: 'briefing', label: 'Briefings' },
  { id: 'do', label: 'Requests & actions' },
  { id: 'watch', label: 'Watches' },
];

const ACTIVE_DELIVERY_KINDS = new Set(['briefing', 'prompt_fire', 'verb_fire', 'deadline']);

const PURPOSES = [
  {
    kind: 'briefing',
    group: 'briefing',
    label: 'Briefing',
    icon: '📰',
    blurb: 'A digest across several topics, synthesized and delivered at a set time.',
    fields: [
      { type: 'chips', key: 'topics', label: 'Topics', placeholder: 'add a topic + Enter (e.g. world news)', required: true,
        hint: 'Each becomes a section. Order is the order you add them.' },
      { type: 'chips', key: 'search_queries', label: 'Refine searches (optional)', placeholder: 'add a search query + Enter',
        hint: 'One per topic, same order — used as the actual web search instead of the topic label. Handy when a topic like "check my coins" needs a real query like "bitcoin ethereum price today".' },
      { type: 'text', key: 'location', label: 'Location', placeholder: 'e.g. Portland, OR (optional)',
        hint: 'Appended to local topics like weather/news. Leave blank for global topics.' },
      { type: 'toggleset', key: 'gather_tools', label: 'Also gather', options: [
        { value: 'weather', label: 'Weather' },
        { value: 'image_search', label: 'Images' },
        { value: 'youtube', label: 'Video' },
      ] },
      { type: 'number', key: 'max_per_topic', label: 'Items per topic', min: 1, max: 5, default: 1 },
      { type: 'bool', key: 'read_aloud', label: 'Read aloud when I open it',
        hint: 'Plays the briefing through your TTS voice. Needs a voice configured in Settings.' },
    ],
    schedule: 'time',
    scheduleDefault: { mode: 'daily', time: '08:00' },
  },
  {
    kind: 'deadline',
    group: 'do',
    label: 'Deadline countdown',
    icon: '⏳',
    blurb: 'Counts down to a date with reminders at set lead times, then retires itself.',
    fields: [
      { type: 'date', key: 'target_date', label: 'Deadline date', required: true,
        hint: 'Reminders fire on a countdown toward this date.' },
      { type: 'time', key: 'local_time', label: 'Remind me at', default: '09:00' },
      { type: 'toggleset', key: 'offsets_days', label: 'Days before to remind', options: [
        { value: '30', label: '30' },
        { value: '14', label: '14' },
        { value: '7', label: '7' },
        { value: '3', label: '3' },
        { value: '1', label: '1' },
        { value: '0', label: 'day-of' },
      ] },
      { type: 'chips', key: 'checklist', label: 'Still to do (optional)', placeholder: 'add an item + Enter' },
      { type: 'textarea', key: 'note', label: 'Note (optional)', rows: 2 },
    ],
    schedule: 'none',
  },
  {
    kind: 'prompt_fire',
    group: 'do',
    label: 'Scheduled request',
    icon: '💬',
    blurb: 'Run a request at a time — it gathers and answers, then delivers the result.',
    fields: [
      { type: 'textarea', key: 'prompt', label: 'Request', rows: 3, required: true,
        placeholder: 'e.g. Summarize today\'s AI research highlights and tell me what matters.' },
    ],
    schedule: 'time',
    scheduleDefault: { mode: 'daily', time: '08:00' },
  },
  {
    kind: 'verb_fire',
    group: 'do',
    label: 'Scheduled action',
    icon: '⚡',
    blurb: 'Fire an app action at a time (e.g. weather.today, music controls). Reversible actions only.',
    fields: [
      { type: 'text', key: 'verb', label: 'Action id', placeholder: 'e.g. weather.today', required: true,
        hint: 'A registered verb id. Only low-stakes, reversible actions are allowed.' },
      { type: 'keyvalue', key: 'verb_args', label: 'Arguments', addLabel: 'add argument' },
    ],
    schedule: 'time',
    scheduleDefault: { mode: 'daily', time: '09:00' },
  },
  {
    kind: 'url_watch',
    group: 'watch',
    label: 'Watch a page',
    icon: '🔎',
    blurb: 'Check a page on a cadence and surface changes — or only when a number crosses a threshold.',
    fields: [
      { type: 'text', key: 'url', label: 'URL', placeholder: 'https://example.com/page', required: true },
      { type: 'condition', key: 'condition', label: 'Only when a number…',
        hint: 'Optional. Leave off to be told on any change.' },
      { type: 'text', key: 'extract_hint', label: 'Which number (optional)', placeholder: 'e.g. price, rating, followers',
        hint: 'Helps pick the right number when the page shows several.' },
      { type: 'text', key: 'intent', label: 'What matters', placeholder: 'e.g. only price or availability changes',
        hint: 'Optional plain-words filter — it judges relevance before alerting.' },
    ],
    schedule: 'interval',
    scheduleDefault: { mode: 'interval', interval: 21600 },
  },
  {
    kind: 'metric_watch',
    group: 'watch',
    label: 'Watch a number',
    icon: '📈',
    blurb: 'Track a number — the weather, or any number on a web page (a price, a count, a rating) — and alert when it crosses your threshold.',
    fields: [
      { type: 'select', key: '_metric_source', label: 'Where the number lives', default: 'weather', options: [
        { value: 'weather', label: 'Weather — temperature for a location' },
        { value: 'webpage', label: 'A web page — price, count, rating…' },
      ] },
      { type: 'text', key: '_metric_location', label: 'Location', placeholder: 'e.g. Portland, OR', required: true,
        showWhen: { key: '_metric_source', value: 'weather' },
        hint: 'Sourced from open-meteo.' },
      { type: 'bool', key: '_metric_imperial', label: 'Use °F (imperial)',
        showWhen: { key: '_metric_source', value: 'weather' } },
      { type: 'text', key: 'url', label: 'Page URL', placeholder: 'https://example.com/price-page', required: true,
        showWhen: { key: '_metric_source', value: 'webpage' },
        hint: 'The page that shows the number (e.g. an exchange or product page).' },
      { type: 'text', key: 'extract_hint', label: 'Which number', placeholder: 'e.g. price, BTC, followers',
        showWhen: { key: '_metric_source', value: 'webpage' },
        hint: 'Helps pick the right number when the page shows several.' },
      { type: 'condition', key: 'condition', label: 'Alert when value…' },
      { type: 'text', key: 'intent', label: 'What matters (optional)', placeholder: 'e.g. only big drops',
        showWhen: { key: '_metric_source', value: 'webpage' } },
    ],
    schedule: 'interval',
    scheduleDefault: { mode: 'interval', interval: 21600 },
  },
  {
    kind: 'feed_digest',
    group: 'watch',
    label: 'Topic digest',
    icon: '🗞️',
    blurb: 'Roll up fresh items for a topic into one note on a cadence.',
    fields: [
      { type: 'text', key: 'topic', label: 'Topic', placeholder: 'e.g. rust async runtime', required: true },
      { type: 'number', key: 'max_items', label: 'Items', min: 1, max: 10, default: 3 },
    ],
    schedule: 'interval',
    scheduleDefault: { mode: 'interval', interval: 86400 },
  },
  {
    kind: 'recurring_search',
    group: 'watch',
    label: 'Saved search',
    icon: '🔁',
    blurb: 'Re-run a search on a cadence and surface only what\'s new.',
    fields: [
      { type: 'text', key: 'query', label: 'Query', placeholder: 'e.g. "post-quantum cryptography"', required: true },
      { type: 'number', key: 'max_results', label: 'Results', min: 1, max: 10, default: 3 },
    ],
    schedule: 'interval',
    scheduleDefault: { mode: 'interval', interval: 86400 },
  },
  {
    kind: 'feed_watch',
    group: 'watch',
    label: 'Follow a creator or feed',
    icon: '📡',
    blurb: 'New videos from a YouTube channel, new podcast episodes, blog or Substack posts, subreddit activity — anything with a feed.',
    fields: [
      { type: 'text', key: '_feed_source', label: 'Who or what to follow', required: true,
        placeholder: 'YouTube channel or @handle · r/subreddit · blog/podcast URL',
        hint: 'Paste it how you know it — a channel link, @handle, r/name, or any site/feed URL. It gets resolved and checked when you save.' },
      { type: 'number', key: 'max_items', label: 'Items per alert', min: 1, max: 10, default: 3 },
    ],
    schedule: 'interval',
    scheduleDefault: { mode: 'interval', interval: 14400 },
    // Resolve the pasted source into a concrete validated feed at save.
    async prepare(params, showErr) {
      const source = (params._feed_source || '').trim();
      delete params._feed_source;
      if (!source) { showErr('Tell me who to follow.'); return null; }
      // Unchanged on edit → keep the already-resolved feed.
      if (params.source_input === source && params.feed_url) return params;
      const res = await _resolveFeed(source);
      if (!res.ok) {
        showErr(res.error || 'Couldn\'t find a feed for that.');
        return null;
      }
      params.feed_url = res.feed_url;
      params.source_label = res.label || source;
      params.source_input = source;
      return params;
    },
  },
  {
    kind: 'github_releases',
    group: 'watch',
    label: 'GitHub releases',
    icon: '🏷️',
    blurb: 'Watch a repository and surface new releases.',
    fields: [
      { type: 'text', key: 'repo', label: 'Repository', placeholder: 'owner/name', required: true },
    ],
    schedule: 'interval',
    scheduleDefault: { mode: 'interval', interval: 86400 },
  },
];

function _purposeForKind(kind) {
  return PURPOSES.find((p) => p.kind === kind) || null;
}

// ── Time / interval helpers ─────────────────────────────────────────

const _INTERVAL_OPTIONS = [
  { label: 'every 15 minutes', seconds: 900 },
  { label: 'every hour', seconds: 3600 },
  { label: 'every 6 hours', seconds: 21600 },
  { label: 'every 12 hours', seconds: 43200 },
  { label: 'every day', seconds: 86400 },
  { label: 'every 3 days', seconds: 259200 },
  { label: 'every week', seconds: 604800 },
];

const _WEEKDAYS = [
  { v: 1, label: 'Mon' }, { v: 2, label: 'Tue' }, { v: 3, label: 'Wed' },
  { v: 4, label: 'Thu' }, { v: 5, label: 'Fri' }, { v: 6, label: 'Sat' },
  { v: 7, label: 'Sun' },
];

function _fmtTime12(hhmm) {
  if (!hhmm || !hhmm.includes(':')) return hhmm || '';
  const [h, m] = hhmm.split(':').map((x) => parseInt(x, 10));
  if (Number.isNaN(h)) return hhmm;
  const am = h < 12;
  const h12 = h % 12 === 0 ? 12 : h % 12;
  return `${h12}:${String(m || 0).padStart(2, '0')} ${am ? 'AM' : 'PM'}`;
}

function _intervalLabel(seconds) {
  const found = _INTERVAL_OPTIONS.find((o) => o.seconds === seconds);
  if (found) return found.label;
  if (seconds < 3600) return `every ${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `every ${Math.round(seconds / 3600)}h`;
  if (seconds < 604800) return `every ${Math.round(seconds / 86400)}d`;
  return `every ${Math.round(seconds / 604800)}w`;
}

// Human schedule summary from a task's params + interval.
function _scheduleSummary(params, interval) {
  const p = params || {};
  if (p.cron && typeof p.cron === 'string') {
    // Engine precedence: cron wins over local_time. Gloss the common
    // shapes locally; anything else shows the raw expression (the
    // builder's live preview is where the full server gloss lives).
    return _cronGloss(p.cron);
  }
  const lt = p.local_time;
  if (lt && typeof lt === 'string' && lt.includes(':')) {
    const t = _fmtTime12(lt);
    if (p.one_shot && p.date) return `Once on ${_esc(p.date)} at ${t}`;
    const days = Array.isArray(p.weekdays) ? p.weekdays.filter((d) => d >= 1 && d <= 7) : [];
    if (!days.length) return `Daily at ${t}`;
    if (days.length === 5 && [1, 2, 3, 4, 5].every((d) => days.includes(d))) return `Weekdays at ${t}`;
    if (days.length === 7) return `Daily at ${t}`;
    const names = _WEEKDAYS.filter((w) => days.includes(w.v)).map((w) => w.label).join(', ');
    return `${names} at ${t}`;
  }
  return _intervalLabel(interval || 86400).replace(/^every/, 'Every');
}

// Client-side gloss for the most common cron shapes — display only
// (validation and the authoritative description come from the server's
// cron-preview endpoint). Unknown shapes render as `Cron: <expr>`.
function _cronGloss(expr) {
  const parts = String(expr).trim().split(/\s+/);
  if (parts.length === 5) {
    const [min, hr, dom, mon, dow] = parts;
    const t = (h, m) => _fmtTime12(`${h.padStart(2, '0')}:${m.padStart(2, '0')}`);
    if (/^\d+$/.test(min) && /^\d+$/.test(hr) && dom === '*' && mon === '*' && dow === '*') {
      return `Daily at ${t(hr, min)}`;
    }
    if (/^\d+$/.test(min) && /^\*\/\d+$/.test(hr) && dom === '*' && mon === '*' && dow === '*') {
      return `Every ${hr.slice(2)} hours`;
    }
    if (min === '0' && hr === '*' && dom === '*' && mon === '*' && dow === '*') {
      return 'Hourly';
    }
    if (/^\d+$/.test(min) && /^\d+$/.test(hr) && /^\d+$/.test(dom) && mon === '*' && dow === '*') {
      return `Monthly on the ${dom} at ${t(hr, min)}`;
    }
  }
  return `Cron: ${expr}`;
}

function _formatRelativeTime(iso) {
  if (!iso) return '—';
  const norm = iso.includes('T') ? iso : iso.replace(' ', 'T') + 'Z';
  const t = Date.parse(norm);
  if (!Number.isFinite(t)) return iso;
  const secs = (Date.now() - t) / 1000;
  if (secs < 0) {
    const f = -secs;
    if (f < 60) return 'in <1m';
    if (f < 3600) return `in ${Math.round(f / 60)}m`;
    if (f < 86400) return `in ${Math.round(f / 3600)}h`;
    return `in ${Math.round(f / 86400)}d`;
  }
  if (secs < 60) return 'just now';
  if (secs < 3600) return `${Math.round(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.round(secs / 3600)}h ago`;
  return `${Math.round(secs / 86400)}d ago`;
}

// ── Field rendering / reading (schema-driven) ───────────────────────

function _fieldHtml(f, value) {
  const id = `sf-${f.key}`;
  const hint = f.hint ? `<div class="sched-field-hint">${_esc(f.hint)}</div>` : '';
  const req = f.required ? '<span class="sched-req">*</span>' : '';
  let control = '';
  switch (f.type) {
    case 'text':
      control = `<input type="text" id="${id}" class="sched-input" data-key="${_esc(f.key)}"
                  placeholder="${_esc(f.placeholder || '')}" value="${_esc(value || '')}" autocomplete="off">`;
      break;
    case 'date':
      // Reads back via the default text path (input.value === 'YYYY-MM-DD').
      control = `<input type="date" id="${id}" class="sched-input sched-date-field" data-key="${_esc(f.key)}"
                  value="${_esc(value || '')}">`;
      break;
    case 'time':
      // Reads back via the default text path (input.value === 'HH:MM').
      control = `<input type="time" id="${id}" class="sched-input sched-time-field" data-key="${_esc(f.key)}"
                  value="${_esc(value || f.default || '09:00')}">`;
      break;
    case 'textarea':
      control = `<textarea id="${id}" class="sched-input sched-textarea" data-key="${_esc(f.key)}"
                  rows="${f.rows || 3}" placeholder="${_esc(f.placeholder || '')}">${_esc(value || '')}</textarea>`;
      break;
    case 'number': {
      const v = (value != null && value !== '') ? value : (f.default ?? '');
      control = `<input type="number" id="${id}" class="sched-input sched-number" data-key="${_esc(f.key)}"
                  min="${f.min ?? 0}" max="${f.max ?? 999}" value="${_esc(String(v))}">`;
      break;
    }
    case 'select': {
      const cur = (value != null && value !== '') ? String(value) : (f.default ?? '');
      const opts = (f.options || []).map((o) =>
        `<option value="${_esc(o.value)}" ${cur === o.value ? 'selected' : ''}>${_esc(o.label)}</option>`).join('');
      control = `<select id="${id}" class="sched-input sched-select" data-key="${_esc(f.key)}">${opts}</select>`;
      break;
    }
    case 'bool': {
      const on = value === true;
      control = `<label class="sched-check"><input type="checkbox" data-key="${_esc(f.key)}" ${on ? 'checked' : ''}>
                  <span>${_esc(f.label)}</span></label>`;
      // bool renders its own label inline; return early without the row label.
      return `<div class="sched-field sched-field-bool"${_whenAttrs(f)}>${control}${hint}</div>`;
    }
    case 'chips': {
      const items = Array.isArray(value) ? value : [];
      const chips = items.map((it, i) => _chipHtml(it, i)).join('');
      control = `<div class="sched-chips" data-key="${_esc(f.key)}">
          <div class="sched-chip-list">${chips}</div>
          <input type="text" class="sched-input sched-chip-input" placeholder="${_esc(f.placeholder || 'add + Enter')}" autocomplete="off">
        </div>`;
      break;
    }
    case 'toggleset': {
      const sel = Array.isArray(value) ? value : [];
      const opts = (f.options || []).map((o) =>
        `<button type="button" class="sched-toggle ${sel.includes(o.value) ? 'on' : ''}"
           data-val="${_esc(o.value)}">${_esc(o.label)}</button>`).join('');
      control = `<div class="sched-toggleset" data-key="${_esc(f.key)}">${opts}</div>`;
      break;
    }
    case 'condition': {
      const c = value && typeof value === 'object' ? value : {};
      const ops = ['<', '<=', '>', '>=', '=='];
      const opSel = ops.map((o) => `<option value="${o}" ${c.op === o ? 'selected' : ''}>${o}</option>`).join('');
      control = `<div class="sched-condition" data-key="${_esc(f.key)}">
          <select class="sched-input sched-cond-op">${opSel}</select>
          <input type="number" step="any" class="sched-input sched-cond-val" placeholder="value"
                 value="${c.value != null ? _esc(String(c.value)) : ''}">
          <input type="text" class="sched-input sched-cond-unit" placeholder="unit (USD, °F…)"
                 value="${_esc(c.unit || '')}">
        </div>`;
      break;
    }
    case 'keyvalue': {
      const obj = value && typeof value === 'object' ? value : {};
      const rows = Object.entries(obj).map(([k, v]) => _kvRowHtml(k, v)).join('');
      control = `<div class="sched-kv" data-key="${_esc(f.key)}">
          <div class="sched-kv-rows">${rows}</div>
          <button type="button" class="sched-kv-add">+ ${_esc(f.addLabel || 'add')}</button>
        </div>`;
      break;
    }
    default:
      control = '';
  }
  return `<div class="sched-field"${_whenAttrs(f)}>
      <label class="sched-field-label" for="${id}">${_esc(f.label)} ${req}</label>
      ${control}${hint}
    </div>`;
}

// Conditional visibility: a field with showWhen renders only while the
// controlling select holds the given value (wired in _wireFields).
function _whenAttrs(f) {
  if (!f.showWhen) return '';
  return ` data-when-key="${_esc(f.showWhen.key)}" data-when-val="${_esc(f.showWhen.value)}"`;
}

function _chipHtml(text, i) {
  return `<span class="sched-chip" data-i="${i}">${_esc(text)}<button type="button" class="sched-chip-x" aria-label="remove">×</button></span>`;
}

function _kvRowHtml(k, v) {
  return `<div class="sched-kv-row">
      <input type="text" class="sched-input sched-kv-k" placeholder="name" value="${_esc(k || '')}">
      <input type="text" class="sched-input sched-kv-v" placeholder="value" value="${_esc(v != null ? String(v) : '')}">
      <button type="button" class="sched-kv-x" aria-label="remove">×</button>
    </div>`;
}

// Wire dynamic field behaviors (chips, toggles, keyvalue) after render.
function _wireFields(scope) {
  // Conditional visibility (showWhen) driven by select fields.
  const updateWhen = () => {
    scope.querySelectorAll('[data-when-key]').forEach((row) => {
      const src = scope.querySelector(`select[data-key="${row.dataset.whenKey}"]`);
      row.style.display = (src && src.value === row.dataset.whenVal) ? '' : 'none';
    });
  };
  scope.querySelectorAll('select[data-key]').forEach((el) =>
    el.addEventListener('change', updateWhen));
  updateWhen();
  // Chips
  scope.querySelectorAll('.sched-chips').forEach((wrap) => {
    const input = wrap.querySelector('.sched-chip-input');
    const list = wrap.querySelector('.sched-chip-list');
    const add = () => {
      const v = (input.value || '').trim();
      if (!v) return;
      const span = document.createElement('span');
      span.className = 'sched-chip';
      span.innerHTML = `${_esc(v)}<button type="button" class="sched-chip-x" aria-label="remove">×</button>`;
      list.appendChild(span);
      input.value = '';
    };
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ',') { e.preventDefault(); add(); }
    });
    input.addEventListener('blur', add);
    list.addEventListener('click', (e) => {
      if (e.target.classList.contains('sched-chip-x')) e.target.closest('.sched-chip')?.remove();
    });
  });
  // Toggle sets
  scope.querySelectorAll('.sched-toggleset .sched-toggle').forEach((btn) => {
    btn.addEventListener('click', () => btn.classList.toggle('on'));
  });
  // Condition op default
  scope.querySelectorAll('.sched-condition').forEach((c) => {
    const op = c.querySelector('.sched-cond-op');
    if (op && !op.value) op.value = '<';
  });
  // Key-value rows
  scope.querySelectorAll('.sched-kv').forEach((wrap) => {
    const rows = wrap.querySelector('.sched-kv-rows');
    wrap.querySelector('.sched-kv-add')?.addEventListener('click', () => {
      const div = document.createElement('div');
      div.innerHTML = _kvRowHtml('', '');
      rows.appendChild(div.firstElementChild);
    });
    rows.addEventListener('click', (e) => {
      if (e.target.classList.contains('sched-kv-x')) e.target.closest('.sched-kv-row')?.remove();
    });
  });
}

// Read a purpose's fields back into a params object.
function _readFields(scope, purpose) {
  const params = {};
  const selVal = (key) => scope.querySelector(`select[data-key="${key}"]`)?.value ?? '';
  for (const f of purpose.fields) {
    // Hidden-by-showWhen fields don't contribute values.
    if (f.showWhen && selVal(f.showWhen.key) !== f.showWhen.value) continue;
    if (f.type === 'condition') {
      const c = scope.querySelector(`.sched-condition[data-key="${f.key}"]`);
      if (c) {
        const valStr = c.querySelector('.sched-cond-val')?.value.trim();
        if (valStr !== '' && valStr != null && !Number.isNaN(parseFloat(valStr))) {
          const cond = { op: c.querySelector('.sched-cond-op')?.value || '<', value: parseFloat(valStr) };
          const unit = c.querySelector('.sched-cond-unit')?.value.trim();
          if (unit) cond.unit = unit;
          params[f.key] = cond;
        }
      }
      continue;
    }
    if (f.type === 'chips') {
      const wrap = scope.querySelector(`.sched-chips[data-key="${f.key}"]`);
      const items = wrap ? Array.from(wrap.querySelectorAll('.sched-chip')).map(
        (c) => c.childNodes[0]?.textContent?.trim()).filter(Boolean) : [];
      params[f.key] = items;
      continue;
    }
    if (f.type === 'toggleset') {
      const wrap = scope.querySelector(`.sched-toggleset[data-key="${f.key}"]`);
      const on = wrap ? Array.from(wrap.querySelectorAll('.sched-toggle.on')).map((b) => b.dataset.val) : [];
      if (on.length) params[f.key] = on;
      continue;
    }
    if (f.type === 'keyvalue') {
      const wrap = scope.querySelector(`.sched-kv[data-key="${f.key}"]`);
      const obj = {};
      wrap?.querySelectorAll('.sched-kv-row').forEach((r) => {
        const k = r.querySelector('.sched-kv-k')?.value.trim();
        const v = r.querySelector('.sched-kv-v')?.value.trim();
        if (k) obj[k] = v;
      });
      if (Object.keys(obj).length) params[f.key] = obj;
      continue;
    }
    if (f.type === 'bool') {
      const el = scope.querySelector(`input[type="checkbox"][data-key="${f.key}"]`);
      if (el && el.checked) params[f.key] = true;
      continue;
    }
    if (f.type === 'number') {
      const el = scope.querySelector(`[data-key="${f.key}"]`);
      if (el && el.value !== '') params[f.key] = parseInt(el.value, 10);
      continue;
    }
    // text / textarea
    const el = scope.querySelector(`[data-key="${f.key}"]`);
    const v = el ? (el.value || '').trim() : '';
    if (v) params[f.key] = v;
  }
  // Special-case metric_watch: the source select decides the shape.
  // Weather folds the _metric_* helpers into `metric` (open-meteo);
  // webpage keeps url/extract_hint/intent/condition — _saveForm maps
  // that to the url_watch kind, whose runner already extracts numbers
  // from pages and applies the condition.
  if (purpose.kind === 'metric_watch') {
    const loc = params._metric_location;
    const imperial = params._metric_imperial === true;
    delete params._metric_location;
    delete params._metric_imperial;
    if ((params._metric_source || 'weather') === 'weather' && loc) {
      params.metric = { provider: 'open_meteo', location: loc, imperial };
    }
  }
  return params;
}

// Pre-fill a purpose's fields from an existing task's params (for edit).
function _fieldValues(purpose, params) {
  const p = params || {};
  const out = {};
  for (const f of purpose.fields) out[f.key] = p[f.key];
  if (purpose.kind === 'metric_watch') {
    out._metric_source = 'weather'; // existing rows are open-meteo; kind is immutable
    const m = p.metric;
    if (m && typeof m === 'object') {
      out._metric_location = m.location || '';
      out._metric_imperial = m.imperial === true;
    } else if (typeof m === 'string') {
      out._metric_location = m;
    }
  }
  if (purpose.kind === 'feed_watch') {
    // Edit shows what the user originally pasted; prepare() skips
    // re-resolution when it's unchanged.
    out._feed_source = p.source_input || p.feed_url || '';
  }
  return out;
}

function _validateRequired(purpose, params) {
  for (const f of purpose.fields) {
    if (!f.required) continue;
    // A field hidden by its showWhen condition isn't required.
    if (f.showWhen && (params[f.showWhen.key] ?? '') !== f.showWhen.value) continue;
    const v = (purpose.kind === 'metric_watch' && f.key === '_metric_location')
      ? (params.metric && params.metric.location) : params[f.key];
    if (f.type === 'chips') { if (!Array.isArray(v) || !v.length) return f.label; }
    else if (v == null || v === '') return f.label;
  }
  return null;
}

// ── Scheduling block ────────────────────────────────────────────────

function _scheduleBlockHtml(purpose, params, interval) {
  // Some purposes carry their own timing in fields (e.g. deadline's
  // target_date + offsets) and need no standard schedule block. _readSchedule
  // returns sane defaults when the block is absent.
  if (purpose.schedule === 'none') return '';
  const p = params || {};
  // Derive current mode from params for edit; else use purpose default.
  let mode = purpose.scheduleDefault?.mode || (purpose.schedule === 'time' ? 'daily' : 'interval');
  let time = purpose.scheduleDefault?.time || '08:00';
  let days = [];
  let date = '';
  let cron = '';
  let intv = interval || purpose.scheduleDefault?.interval || 86400;
  if (p.cron && typeof p.cron === 'string') {
    // Cron wins over local_time (engine precedence) — edit reopens in
    // custom mode with the expression intact.
    mode = 'cron';
    cron = p.cron;
    if (p.local_time && p.local_time.includes(':')) time = p.local_time;
  } else if (p.local_time && p.local_time.includes(':')) {
    time = p.local_time;
    const wd = Array.isArray(p.weekdays) ? p.weekdays.filter((d) => d >= 1 && d <= 7) : [];
    if (p.one_shot && p.date) { mode = 'once'; date = p.date; }
    else if (wd.length === 5 && [1, 2, 3, 4, 5].every((d) => wd.includes(d))) { mode = 'weekdays'; }
    else if (wd.length && !(wd.length === 7)) { mode = 'weekly'; days = wd; }
    else { mode = 'daily'; }
  } else if (interval) {
    mode = 'interval';
  }

  const modeOpts = [
    { v: 'daily', label: 'Every day' },
    { v: 'weekdays', label: 'Weekdays (Mon–Fri)' },
    { v: 'weekly', label: 'Specific days' },
    { v: 'interval', label: 'Every…' },
    { v: 'once', label: 'Once (a date)' },
    { v: 'cron', label: 'Custom (cron)' },
  ].map((o) => `<option value="${o.v}" ${mode === o.v ? 'selected' : ''}>${o.label}</option>`).join('');

  const intvOpts = _INTERVAL_OPTIONS.map(
    (o) => `<option value="${o.seconds}" ${intv === o.seconds ? 'selected' : ''}>${o.label}</option>`).join('');

  const dayChips = _WEEKDAYS.map((w) =>
    `<button type="button" class="sched-day ${days.includes(w.v) ? 'on' : ''}" data-day="${w.v}">${w.label}</button>`).join('');

  return `<div class="sched-schedule" data-mode="${mode}">
      <label class="sched-field-label">Schedule</label>
      <div class="sched-sched-row">
        <select class="sched-input sched-mode">${modeOpts}</select>
        <input type="time" class="sched-input sched-time" value="${_esc(time)}">
        <select class="sched-input sched-interval">${intvOpts}</select>
        <input type="date" class="sched-input sched-date" value="${_esc(date)}">
        <input type="text" class="sched-input sched-cron" value="${_esc(cron)}"
               placeholder="0 */2 * * *" spellcheck="false" autocomplete="off"
               title="minute hour day month weekday — e.g. '0 9 * * mon-fri', '@hourly'">
      </div>
      <div class="sched-days">${dayChips}</div>
      <div class="sched-cron-preview" aria-live="polite"></div>
      <div class="sched-sched-summary"></div>
    </div>`;
}

function _wireSchedule(scope) {
  const block = scope.querySelector('.sched-schedule');
  if (!block) return;
  const modeSel = block.querySelector('.sched-mode');
  const cronInput = block.querySelector('.sched-cron');
  const cronOut = block.querySelector('.sched-cron-preview');

  // Debounced live preview: server-validated gloss + the next 3 fire
  // times in the user's timezone, updating as they type.
  let cronTimer = null;
  let cronSeq = 0;
  const refreshCronPreview = () => {
    if (!cronOut) return;
    const expr = (cronInput?.value || '').trim();
    if (modeSel.value !== 'cron' || !expr) { cronOut.textContent = ''; cronOut.classList.remove('bad'); return; }
    if (cronTimer) clearTimeout(cronTimer);
    const seq = ++cronSeq;
    cronTimer = setTimeout(async () => {
      const res = await _cronPreview(expr);
      if (seq !== cronSeq) return; // a newer keystroke superseded us
      if (!res.ok) {
        cronOut.textContent = `✕ ${res.error || 'invalid expression'}`;
        cronOut.classList.add('bad');
        return;
      }
      cronOut.classList.remove('bad');
      const fires = (res.next_fires || []).join('  ·  ');
      cronOut.textContent = `✓ ${res.description}${fires ? ` — next: ${fires}` : ''}${res.timezone ? ` (${res.timezone})` : ''}`;
    }, 350);
  };

  const updateVis = () => {
    const mode = modeSel.value;
    block.dataset.mode = mode;
    block.querySelector('.sched-time').style.display = (mode === 'interval' || mode === 'cron') ? 'none' : '';
    block.querySelector('.sched-interval').style.display = (mode === 'interval') ? '' : 'none';
    block.querySelector('.sched-date').style.display = (mode === 'once') ? '' : 'none';
    block.querySelector('.sched-days').style.display = (mode === 'weekly') ? '' : 'none';
    if (cronInput) cronInput.style.display = (mode === 'cron') ? '' : 'none';
    if (cronOut) cronOut.style.display = (mode === 'cron') ? '' : 'none';
    refreshCronPreview();
    _updateScheduleSummary(scope);
  };
  modeSel.addEventListener('change', updateVis);
  block.querySelectorAll('input, select').forEach((el) =>
    el.addEventListener('input', () => _updateScheduleSummary(scope)));
  if (cronInput) cronInput.addEventListener('input', refreshCronPreview);
  block.querySelectorAll('.sched-day').forEach((b) =>
    b.addEventListener('click', () => { b.classList.toggle('on'); _updateScheduleSummary(scope); }));
  updateVis();
}

function _readSchedule(scope) {
  const block = scope.querySelector('.sched-schedule');
  if (!block) return { params: {}, interval: 86400 };
  const mode = block.querySelector('.sched-mode').value;
  const time = block.querySelector('.sched-time').value || '08:00';
  const date = block.querySelector('.sched-date').value || '';
  const interval = parseInt(block.querySelector('.sched-interval').value, 10) || 86400;
  const days = Array.from(block.querySelectorAll('.sched-day.on')).map((b) => parseInt(b.dataset.day, 10));
  const params = {};
  let intervalSeconds = 86400;
  if (mode === 'cron') {
    const cron = (block.querySelector('.sched-cron')?.value || '').trim();
    if (cron) params.cron = cron;
  } else if (mode === 'interval') {
    intervalSeconds = interval;
  } else {
    params.local_time = time;
    if (mode === 'weekdays') params.weekdays = [1, 2, 3, 4, 5];
    else if (mode === 'weekly') params.weekdays = days.length ? days : [];
    else if (mode === 'once') { params.one_shot = true; if (date) params.date = date; }
  }
  return { params, interval: intervalSeconds };
}

function _updateScheduleSummary(scope) {
  const out = scope.querySelector('.sched-sched-summary');
  if (!out) return;
  const { params, interval } = _readSchedule(scope);
  out.textContent = `→ ${_scheduleSummary(params, interval)}`;
}

// ── List rendering ──────────────────────────────────────────────────

function _statusFor(t) {
  if (!t.enabled) return { cls: 'paused', label: 'Paused' };
  if (t.consecutive_error_count >= 5) return { cls: 'error', label: 'Auto-paused (errors)' };
  if (t.consecutive_error_count > 0) return { cls: 'warn', label: `${t.consecutive_error_count} recent error${t.consecutive_error_count === 1 ? '' : 's'}` };
  return { cls: 'ok', label: 'Active' };
}

// Effective delivery for a task: the user's explicit params.delivery
// choice wins; otherwise the kind default. Mirrors the server's
// _surface_importance resolution exactly.
function _deliveryFor(kind, params) {
  const pref = params && params.delivery;
  if (pref === 'alert' || pref === 'quiet') return pref;
  return ACTIVE_DELIVERY_KINDS.has(kind) ? 'alert' : 'quiet';
}

function _deliveryChip(kind, params) {
  if (_deliveryFor(kind, params) === 'alert') {
    return `<span class="sched-delivery sched-delivery-loud" title="Alerts every device when it fires (even with a tab open)">🔔 Alerts all devices</span>`;
  }
  return `<span class="sched-delivery sched-delivery-quiet" title="Chimes in an open tab; only pushes to a device when you're away">🔕 Quiet digest</span>`;
}

const _DELIVERY_NOTES = {
  alert: 'When this fires it notifies every device and plays a sound — even with a tab open.',
  quiet: 'Chimes in an open tab; pushes to a device only when you\'re away.',
};

// Interactive delivery choice for the form — the user picks, never the
// kind alone. Defaults to the kind's convention; the choice persists as
// params.delivery so it survives default changes.
function _deliveryPickerHtml(kind, params) {
  const cur = _deliveryFor(kind, params);
  const seg = (v, icon, label) =>
    `<button type="button" class="sched-toggle sched-seg ${cur === v ? 'on' : ''}" data-dv="${v}">${icon} ${label}</button>`;
  return `<div class="sched-field sched-delivery-field">
      <label class="sched-field-label">Delivery</label>
      <div class="sched-delivery-seg" data-delivery="${cur}">
        ${seg('alert', '🔔', 'Alert all devices')}
        ${seg('quiet', '🔕', 'Quiet digest')}
      </div>
      <div class="sched-field-hint sched-delivery-note">${_DELIVERY_NOTES[cur]}</div>
    </div>`;
}

function _wireDelivery(scope) {
  const seg = scope.querySelector('.sched-delivery-seg');
  if (!seg) return;
  seg.querySelectorAll('.sched-seg').forEach((b) => b.addEventListener('click', () => {
    seg.querySelectorAll('.sched-seg').forEach((x) => x.classList.toggle('on', x === b));
    seg.dataset.delivery = b.dataset.dv;
    const note = scope.querySelector('.sched-delivery-note');
    if (note) note.textContent = _DELIVERY_NOTES[b.dataset.dv] || '';
  }));
}

function _readDelivery(scope) {
  const seg = scope.querySelector('.sched-delivery-seg');
  const v = seg?.dataset.delivery;
  return (v === 'alert' || v === 'quiet') ? v : null;
}

function _taskCardHtml(t) {
  const purpose = _purposeForKind(t.kind);
  const label = purpose ? purpose.label : t.kind;
  const icon = purpose ? purpose.icon : '•';
  const status = _statusFor(t);
  const schedule = _scheduleSummary(t.params, t.interval_seconds);
  const summary = t.last_result_summary
    ? `<div class="sched-card-summary">${_esc(t.last_result_summary)}</div>` : '';
  const errLine = (t.consecutive_error_count > 0 && t.last_error)
    ? `<div class="sched-card-error" title="${_esc(t.last_error)}">⚠ ${_esc(t.last_error)}</div>` : '';
  return `
    <div class="sched-card ${!t.enabled ? 'sched-card-paused' : ''}" data-id="${_esc(String(t.id))}">
      <div class="sched-card-head">
        <span class="sched-card-icon">${icon}</span>
        <span class="sched-card-title">${_esc(t.title)}</span>
        <span class="sched-card-kind">${_esc(label)}</span>
        <span class="sched-status sched-status-${status.cls}" title="${_esc(status.label)}"></span>
      </div>
      <div class="sched-card-sched">
        <span class="sched-card-clock">🕑 ${_esc(schedule)}</span>
        ${_deliveryChip(t.kind, t.params)}
      </div>
      <div class="sched-card-meta">
        <span>last ${_esc(_formatRelativeTime(t.last_run_at))}</span>
        <span>·</span>
        <span>next ${t.enabled ? _esc(_formatRelativeTime(t.next_run_at)) : 'paused'}</span>
      </div>
      ${summary}
      ${errLine}
      <div class="sched-card-actions">
        <button type="button" class="sched-btn" data-act="run">Run now</button>
        <button type="button" class="sched-btn" data-act="toggle">${t.enabled ? 'Pause' : 'Resume'}</button>
        <button type="button" class="sched-btn" data-act="edit">Edit</button>
        <button type="button" class="sched-btn sched-btn-history" data-act="history">History</button>
        <button type="button" class="sched-btn sched-btn-danger" data-act="remove" aria-label="Delete">Delete</button>
      </div>
      <div class="sched-history" hidden></div>
    </div>`;
}

function _runRowHtml(r) {
  const icons = { fired: '🔔', silent: '·', suppressed: '🤫', error: '⚠', cancelled: '⊘' };
  const icon = icons[r.status] || '·';
  const when = _formatRelativeTime(r.ran_at);
  const judge = r.details && r.details.judge && r.details.judge.reason
    ? `<div class="sched-run-judge">judged: ${_esc(r.details.judge.reason)}</div>` : '';
  const elapsed = r.details && r.details.elapsed_ms != null
    ? `<span class="sched-run-elapsed">${Math.round(r.details.elapsed_ms)}ms</span>` : '';
  return `<div class="sched-run sched-run-${_esc(r.status)}">
      <span class="sched-run-icon">${icon}</span>
      <div class="sched-run-body">
        <div class="sched-run-top"><span class="sched-run-status">${_esc(r.status)}</span>
          <span class="sched-run-when">${_esc(when)}</span>${elapsed}</div>
        ${r.summary ? `<div class="sched-run-summary">${_esc(r.summary)}</div>` : ''}
        ${_runContentHtml(r.details)}
        ${judge}
      </div>
    </div>`;
}

// Full delivered body (briefing sections, hero image, citations) — the
// summary above is only the push-notification headline. Collapsed by
// default behind a "Show full briefing" toggle so the history list
// stays scannable.
function _runContentHtml(details) {
  if (!details || !details.content) return '';
  const hero = details.hero_image_url
    ? `<img class="sched-run-hero" src="${_esc(details.hero_image_url)}" alt="" loading="lazy">` : '';
  const cites = Array.isArray(details.citations) && details.citations.length
    ? `<div class="sched-run-cites">${details.citations.map((c) => {
        const url = typeof c === 'string' ? c : (c.url || '');
        const title = typeof c === 'string' ? c : (c.title || c.url || '');
        return url ? `<a href="${_esc(url)}" target="_blank" rel="noopener">${_esc(title)}</a>` : '';
      }).filter(Boolean).join(' · ')}</div>` : '';
  const body = _esc(details.content).replace(/\n/g, '<br>');
  return `<details class="sched-run-content"><summary>Show full briefing</summary>
      ${hero}<div class="sched-run-content-body">${body}</div>${cites}</details>`;
}

// One-tap starter briefings for the empty state — universal, ready-to-run
// presets (no user input needed). Each instantiates a `briefing` standing
// task via the normal create path; location-aware topics pick up the user's
// saved location automatically. Curated from docs/briefing-use-cases.md.
const STARTER_BRIEFINGS = [
  { icon: '🌍', label: 'World in 5',
    blurb: 'The 5 most important headlines, every morning.',
    params: { topics: ['the 5 most important world news headlines today'], local_time: '07:00' } },
  { icon: '📰', label: 'Daily news briefing',
    blurb: 'National, international & local news.',
    params: { topics: ['top international news', 'US national news', 'local news'], local_time: '07:00' } },
  { icon: '⛅', label: 'Weather at 8',
    blurb: 'Today & tomorrow for where you are.',
    params: { topics: ['weather'], gather_tools: ['weather'], local_time: '08:00' } },
  { icon: '🚗', label: 'Weekday commute',
    blurb: 'Weather + local traffic, Mon–Fri.',
    params: { topics: ['weather', 'local traffic and road conditions'], gather_tools: ['weather'], weekdays: [1, 2, 3, 4, 5], local_time: '07:00' } },
  { icon: '🤖', label: 'Tech & AI roundup',
    blurb: "What's new in tech and AI.",
    params: { topics: ['artificial intelligence news', 'technology industry news'], local_time: '08:00' } },
  { icon: '📈', label: 'Markets open',
    blurb: 'Stocks & crypto before the bell.',
    params: { topics: ['stock market news', 'cryptocurrency news'], local_time: '08:30' } },
  { icon: '🏟️', label: 'Sports headlines',
    blurb: "Last night's results & big stories.",
    params: { topics: ['sports headlines and scores'], local_time: '08:00' } },
  { icon: '🔬', label: 'Science & space',
    blurb: 'Discoveries, research & the cosmos.',
    params: { topics: ['science news', 'space and astronomy news'], local_time: '08:00' } },
  { icon: '🎬', label: 'Screen & stream',
    blurb: 'New releases worth your time.',
    params: { topics: ['new movie and TV releases', 'streaming news'], local_time: '18:00' } },
  { icon: '🎟️', label: 'This weekend nearby',
    blurb: 'Things to do, every Friday.',
    params: { topics: ['local events and things to do this weekend'], gather_tools: ['image_search'], weekdays: [5], local_time: '17:00' } },
  { icon: '💡', label: 'Teach me something',
    blurb: 'One interesting idea a day.',
    params: { topics: ['an interesting concept or fact worth learning today, explained clearly'], local_time: '09:00' } },
  { icon: '🩺', label: 'Health & wellness',
    blurb: 'Practical, well-sourced health news.',
    params: { topics: ['health and wellness news'], local_time: '08:00' } },
];

function _renderList(tasks, topics) {
  if (!_modal) return;
  const listEl = _modal.querySelector('.sched-list');
  if (!listEl) return;

  if (!tasks.length) {
    listEl.innerHTML = `<div class="sched-empty">
        <p>No scheduled tasks yet.</p>
        <p class="sched-empty-sub">Create a morning briefing, a page watch, or a scheduled request — they run on the cadence you set and notify you when they fire.</p>
        <div class="sched-starters">
          <p class="sched-starters-title">Not sure where to start? One tap to set one up:</p>
          <div class="sched-starters-grid">
            ${STARTER_BRIEFINGS.map((s, i) => `
              <button type="button" class="sched-starter" data-starter="${i}">
                <span class="sched-starter-icon" aria-hidden="true">${s.icon}</span>
                <span class="sched-starter-label">${_esc(s.label)}</span>
                <span class="sched-starter-blurb">${_esc(s.blurb)}</span>
              </button>`).join('')}
          </div>
        </div>
      </div>`;
  } else {
    // Group by purpose group.
    const byGroup = {};
    for (const t of tasks) {
      const purpose = _purposeForKind(t.kind);
      const g = purpose ? purpose.group : 'watch';
      (byGroup[g] = byGroup[g] || []).push(t);
    }
    listEl.innerHTML = GROUPS.map((g) => {
      const items = byGroup[g.id] || [];
      if (!items.length) return '';
      return `<div class="sched-group">
          <div class="sched-group-title">${_esc(g.label)} <span class="sched-group-count">${items.length}</span></div>
          ${items.map(_taskCardHtml).join('')}
        </div>`;
    }).join('');
  }
  _wireTaskCards();
  _wireStarters();

  // Topics section (the lightweight watch-list — a separate backend).
  const topicsEl = _modal.querySelector('.sched-topics-list');
  if (topicsEl) {
    if (!topics || !topics.length) {
      topicsEl.innerHTML = `<p class="sched-topics-empty">Nothing tracked. Add a phrase or an RSS URL above.</p>`;
    } else {
      topicsEl.innerHTML = topics.map((t) => {
        const badge = t.feed_url
          ? `<span class="sched-topic-badge" title="${_esc(t.feed_url)}">${_esc(t.feed_kind || 'rss')}</span>` : '';
        return `<div class="sched-topic" data-id="${_esc(String(t.id))}">
            <span class="sched-topic-name">${_esc(t.topic)}</span>${badge}
            <button type="button" class="sched-topic-x" aria-label="Remove">×</button>
          </div>`;
      }).join('');
      topicsEl.querySelectorAll('.sched-topic-x').forEach((btn) => {
        btn.addEventListener('click', async (e) => {
          const row = e.currentTarget.closest('.sched-topic');
          const id = row?.getAttribute('data-id');
          if (!id) return;
          row.classList.add('removing');
          if (await _removeTopic(id)) await _refreshTopics();
          else row.classList.remove('removing');
        });
      });
    }
  }
}

function _wireTaskCards() {
  _modal.querySelectorAll('.sched-card').forEach((card) => {
    const id = card.getAttribute('data-id');
    card.querySelector('[data-act="run"]')?.addEventListener('click', async (e) => {
      const btn = e.currentTarget;
      btn.disabled = true; btn.textContent = 'Running…';
      const res = await _runNow(id);
      btn.disabled = false; btn.textContent = 'Run now';
      if (res && res.ok) {
        // Show the ACTUAL delivered briefing right here — the card only
        // renders the one-line headline, which read as "it only did the
        // weather" when the body had every section all along.
        const result = res.result || {};
        const details = result.details || {};
        const box = document.createElement('div');
        box.className = 'sched-run-result';
        const head = `<div class="sched-run-result-head">${_esc(result.summary || 'Done')}</div>`;
        const bodyHtml = _runContentHtml(details);
        box.innerHTML = head + (bodyHtml || '');
        if (bodyHtml) box.querySelector('.sched-run-content').open = true;
        card.querySelector('.sched-run-result')?.remove();
        card.appendChild(box);
      }
    });
    card.querySelector('[data-act="toggle"]')?.addEventListener('click', async () => {
      const isEnabled = !card.classList.contains('sched-card-paused');
      if ((await _patchTask(id, { enabled: !isEnabled })).ok) await _refreshTasks();
    });
    card.querySelector('[data-act="edit"]')?.addEventListener('click', async () => {
      const { tasks } = await _fetchTasks();
      const task = tasks.find((t) => String(t.id) === String(id));
      if (task) _openForm(task);
    });
    card.querySelector('[data-act="remove"]')?.addEventListener('click', async () => {
      if (!window.confirm('Delete this scheduled task? This cannot be undone.')) return;
      card.classList.add('removing');
      if (await _removeTask(id)) await _refreshTasks();
      else card.classList.remove('removing');
    });
    card.querySelector('[data-act="history"]')?.addEventListener('click', async (e) => {
      const box = card.querySelector('.sched-history');
      const btn = e.currentTarget;
      if (!box.hidden) { box.hidden = true; btn.textContent = 'History'; return; }
      btn.textContent = 'Hide history';
      box.hidden = false;
      box.innerHTML = `<div class="sched-history-loading">loading…</div>`;
      const runs = await _fetchRuns(id);
      box.innerHTML = runs.length
        ? runs.map(_runRowHtml).join('')
        : `<div class="sched-history-empty">No runs yet. Hit “Run now” to test it.</div>`;
    });
  });
}

// ── Form (create / edit) ────────────────────────────────────────────

function _openPicker() {
  _editing = null;
  _view = 'form';
  const panel = _modal.querySelector('.sched-form-panel');
  const grouped = GROUPS.map((g) => {
    const items = PURPOSES.filter((p) => p.group === g.id && (!_kinds.length || _kinds.includes(p.kind)));
    if (!items.length) return '';
    return `<div class="sched-pick-group">
        <div class="sched-pick-group-title">${_esc(g.label)}</div>
        <div class="sched-pick-grid">
          ${items.map((p) => `<button type="button" class="sched-pick" data-kind="${_esc(p.kind)}">
              <span class="sched-pick-icon">${p.icon}</span>
              <span class="sched-pick-label">${_esc(p.label)}</span>
              <span class="sched-pick-blurb">${_esc(p.blurb)}</span>
            </button>`).join('')}
        </div>
      </div>`;
  }).join('');
  panel.innerHTML = `
    <div class="sched-form-head">
      <button type="button" class="sched-back" data-act="back">← Back</button>
      <span class="sched-form-title">New scheduled task</span>
    </div>
    <div class="sched-pick-wrap">${grouped}</div>`;
  panel.querySelector('[data-act="back"]').addEventListener('click', _showList);
  panel.querySelectorAll('.sched-pick').forEach((b) =>
    b.addEventListener('click', () => _openForm(null, b.dataset.kind)));
  _swap();
}

// One-tap starter briefings (empty-state). Creates the briefing via the
// normal create path, then refreshes the list (which replaces the empty
// state with the new task). Errors surface inline on the empty-state hint.
function _wireStarters() {
  if (!_modal) return;
  _modal.querySelectorAll('.sched-starter').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const s = STARTER_BRIEFINGS[parseInt(btn.dataset.starter, 10)];
      if (!s) return;
      btn.disabled = true;
      btn.classList.add('sched-starter-pending');
      const res = await _addTask({
        title: s.label,
        kind: 'briefing',
        params: s.params,
        interval_seconds: 86400,
      });
      if (res && res.ok) {
        await _refreshTasks();
      } else {
        btn.disabled = false;
        btn.classList.remove('sched-starter-pending');
        const reason = res ? res.reason : 'network';
        const hint = _modal.querySelector('.sched-empty-sub');
        if (hint) {
          hint.textContent = reason === 'duplicate'
            ? 'You already have that one — edit it from the list.'
            : `Couldn't set that up (${reason}). Try again.`;
        }
      }
    });
  });
}

function _openForm(task, kindOverride) {
  _editing = task || null;
  _view = 'form';
  const kind = task ? task.kind : kindOverride;
  const purpose = _purposeForKind(kind);
  if (!purpose) { _showList(); return; }
  const params = task ? (task.params || {}) : {};
  const interval = task ? task.interval_seconds : (purpose.scheduleDefault?.interval || 86400);
  const values = _fieldValues(purpose, params);
  const title = task ? task.title : '';

  const fieldsHtml = purpose.fields.map((f) => _fieldHtml(f, values[f.key])).join('');
  const panel = _modal.querySelector('.sched-form-panel');
  panel.innerHTML = `
    <div class="sched-form-head">
      <button type="button" class="sched-back" data-act="back">← Back</button>
      <span class="sched-form-title">${task ? 'Edit' : 'New'} ${_esc(purpose.label.toLowerCase())}</span>
    </div>
    <div class="sched-form-body">
      <div class="sched-form-blurb">${purpose.icon} ${_esc(purpose.blurb)}</div>
      <div class="sched-field">
        <label class="sched-field-label" for="sf-title">Name <span class="sched-req">*</span></label>
        <input type="text" id="sf-title" class="sched-input sched-title-input"
               placeholder="e.g. Morning briefing" value="${_esc(title)}" autocomplete="off">
      </div>
      ${fieldsHtml}
      ${_scheduleBlockHtml(purpose, params, interval)}
      ${_deliveryPickerHtml(kind, params)}
      <div class="sched-form-error" hidden></div>
    </div>
    <div class="sched-form-foot">
      <button type="button" class="sched-btn" data-act="cancel">Cancel</button>
      <button type="button" class="sched-btn sched-btn-primary" data-act="save">${task ? 'Save changes' : 'Create'}</button>
    </div>`;

  _wireFields(panel);
  _wireSchedule(panel);
  _wireDelivery(panel);
  // Kind is immutable on an existing row — lock the source select so an
  // edit can't silently demand a different kind.
  if (task) {
    panel.querySelector('select[data-key="_metric_source"]')
      ?.setAttribute('disabled', '');
  }
  panel.querySelector('[data-act="back"]').addEventListener('click', _showList);
  panel.querySelector('[data-act="cancel"]').addEventListener('click', _showList);
  panel.querySelector('[data-act="save"]').addEventListener('click', () => _saveForm(purpose));
  panel.dataset.kind = kind;
  _swap();
  panel.querySelector('#sf-title')?.focus();
}

async function _saveForm(purpose) {
  const panel = _modal.querySelector('.sched-form-panel');
  const errEl = panel.querySelector('.sched-form-error');
  const showErr = (msg) => { errEl.textContent = msg; errEl.hidden = false; };

  const title = (panel.querySelector('#sf-title')?.value || '').trim();
  if (!title) { showErr('Give it a name.'); return; }

  const kindParams = _readFields(panel, purpose);
  const missing = _validateRequired(purpose, kindParams);
  if (missing) { showErr(`${missing} is required.`); return; }

  // "Watch a number" is one purpose over two kinds: weather stays
  // metric_watch; a number on a web page IS a url_watch (its runner
  // already extracts values + applies the condition).
  let kind = purpose.kind;
  if (purpose.kind === 'metric_watch') {
    const src = kindParams._metric_source || 'weather';
    delete kindParams._metric_source;
    if (src === 'webpage') kind = 'url_watch';
    if (_editing && kind !== _editing.kind) {
      showErr('The source can\'t change on an existing watch — create a new one instead.');
      return;
    }
  }

  // Purpose-level async preparation (e.g. resolving a pasted creator
  // into a validated feed). Runs before persistence; a null return
  // means the purpose already surfaced the error.
  let prepared = kindParams;
  if (typeof purpose.prepare === 'function') {
    const btn = panel.querySelector('[data-act="save"]');
    btn.disabled = true; btn.textContent = 'Checking…';
    try {
      prepared = await purpose.prepare(kindParams, showErr);
    } finally {
      btn.disabled = false;
      btn.textContent = _editing ? 'Save changes' : 'Create';
    }
    if (!prepared) return;
  }

  const sched = _readSchedule(panel);
  const params = { ...prepared, ...sched.params };
  const delivery = _readDelivery(panel);
  if (delivery) params.delivery = delivery;

  const saveBtn = panel.querySelector('[data-act="save"]');
  saveBtn.disabled = true; saveBtn.textContent = 'Saving…';

  let res;
  if (_editing) {
    res = await _patchTask(_editing.id, {
      title, params, interval_seconds: sched.interval,
    });
  } else {
    res = await _addTask({
      title, kind, params, interval_seconds: sched.interval,
    });
  }

  saveBtn.disabled = false; saveBtn.textContent = _editing ? 'Save changes' : 'Create';
  if (res && res.ok) {
    _editing = null;
    await _refreshTasks();
    _showList();
  } else {
    const reason = res ? res.reason : 'network';
    showErr(reason === 'duplicate' ? 'A task with that name already exists.' : `Couldn't save (${reason}).`);
  }
}

// ── View swap ───────────────────────────────────────────────────────

function _swap() {
  if (!_modal) return;
  _modal.querySelector('.sched-list-view').hidden = (_view !== 'list');
  _modal.querySelector('.sched-form-view').hidden = (_view !== 'form');
}

function _showList() {
  _view = 'list';
  _editing = null;
  _swap();
}

// ── Refresh ─────────────────────────────────────────────────────────

let _topicsCache = [];
async function _refreshTopics() {
  _topicsCache = await _fetchTopics();
  const { tasks } = await _fetchTasks();
  _renderList(tasks, _topicsCache);
}
async function _refreshTasks() {
  const { tasks } = await _fetchTasks();
  _renderList(tasks, _topicsCache);
}
async function _refresh() {
  const [{ tasks }, topics] = await Promise.all([_fetchTasks(), _fetchTopics()]);
  _topicsCache = topics;
  _renderList(tasks, topics);
}

// ── Shell ───────────────────────────────────────────────────────────

function _buildModal() {
  if (_modal) return _modal;
  const root = document.createElement('div');
  root.className = 'sched-modal hidden';
  root.innerHTML = `
    <div class="sched-backdrop"></div>
    <div class="sched-panel" role="dialog" aria-label="Schedule">
      <header class="sched-header">
        <span class="sched-title">Schedule</span>
        <button type="button" class="sched-close" aria-label="Close">×</button>
      </header>

      <div class="sched-list-view">
        <div class="sched-toolbar">
          <span class="sched-toolbar-sub">Briefings, requests, and watches your companion runs for you.</span>
          <button type="button" class="sched-btn sched-btn-primary sched-new">+ New</button>
        </div>
        <div class="sched-list" aria-live="polite"><div class="sched-empty">loading…</div></div>

        <section class="sched-topics">
          <div class="sched-topics-head">
            <span class="sched-topics-title">Topics watch-list</span>
          </div>
          <form class="sched-topics-form">
            <input type="text" class="sched-input sched-topics-input"
                   placeholder="topic name or RSS URL" autocomplete="off" aria-label="Add a topic or feed">
            <button type="submit" class="sched-btn">Add</button>
          </form>
          <div class="sched-topics-hint">Bare phrases route to HN/arXiv/etc; paste an RSS URL to subscribe to a feed.</div>
          <div class="sched-topics-list" aria-live="polite"></div>
        </section>
      </div>

      <div class="sched-form-view" hidden>
        <div class="sched-form-panel"></div>
      </div>
    </div>`;
  document.body.appendChild(root);
  _modal = root;

  root.querySelector('.sched-close').addEventListener('click', close);
  root.querySelector('.sched-backdrop').addEventListener('click', close);
  root.querySelector('.sched-new').addEventListener('click', _openPicker);

  // Topics add form
  const tForm = root.querySelector('.sched-topics-form');
  const tInput = root.querySelector('.sched-topics-input');
  tForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const v = (tInput.value || '').trim();
    if (!v) return;
    tInput.disabled = true;
    const res = await _addTopic(v);
    tInput.disabled = false;
    if (res.ok) { tInput.value = ''; await _refreshTopics(); tInput.focus(); }
    else if (res.reason === 'duplicate') {
      tInput.classList.add('shake');
      setTimeout(() => tInput.classList.remove('shake'), 400);
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && _modal && !_modal.classList.contains('hidden')) {
      if (_view === 'form') _showList();
      else close();
    }
  });
  return root;
}

export async function open() {
  const { kinds } = await _fetchTasks();
  _kinds = kinds && kinds.length ? kinds : PURPOSES.map((p) => p.kind);
  _buildModal();
  _view = 'list';
  _swap();
  _modal.classList.remove('hidden');
  // Focus trap + initial focus + focus-restore + aria-modal, live only while
  // shown. escapeCloses:false — the modal keeps its own Escape handler
  // (form view → back to list; list view → close), wired in _buildModal.
  if (_dialog) { try { _dialog.release(); } catch (_) {} _dialog = null; }
  _dialog = installDialog(_modal.querySelector('.sched-panel'), {
    escapeCloses: false,
    initialFocus: '.sched-new',
    setAria: true,   // panel has role/aria-label; this adds aria-modal
  });
  await _refresh();
}

export function close() {
  if (_dialog) { try { _dialog.release(); } catch (_) {} _dialog = null; }
  if (_modal) _modal.classList.add('hidden');
}

// Open straight to a specific standing task's editor — used by the calendar
// surface when a companion (amber) event is clicked, so the full task power
// (schedule builder, run-now, history) is one tap away from the grid.
export async function openTask(taskId) {
  await open();
  try {
    const { tasks } = await _fetchTasks();
    const task = tasks.find((t) => String(t.id) === String(taskId));
    if (task) _openForm(task);
  } catch (_) { /* fall back to the list view already shown by open() */ }
}
