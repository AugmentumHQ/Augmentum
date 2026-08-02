/**
 * calendar/index.js — the Augmentum Calendar surface.
 *
 * A first-class, professional calendar (Month · Week · Day · Agenda) that
 * overlays three layers on one time grid:
 *   • Augmentum  — native, user-owned events (created/edited here; optionally
 *                  mirrored to the user's devices over CalDAV).
 *   • Calendar   — appointments synced FROM connected CalDAV servers.
 *   • Companion  — occurrences of the companion's standing tasks (read-only;
 *                  clicking opens the full task editor).
 *
 * Backend: augmentum/proxy/calendar_routes.py. Persona-agnostic copy — this
 * is shipped chrome. Opened from the header menu + command palette.
 */

import { installDialog } from '../_focus-trap.js';

let _modal = null;
let _dialog = null;
let _view = 'month';          // 'month' | 'week' | 'day' | 'agenda'
let _cursor = null;           // anchor Date (local) the current view is built around
let _events = [];             // last-fetched instances for the visible range
let _layers = { augmentum: true, calendar: true, companion: true };
let _syncAvailable = false;
let _lastSynced = 0;          // epoch seconds of the last CalDAV pull
let _syncing = false;
let _syncTimer = null;        // periodic auto-sync while the modal is open
let _fetchSeq = 0;

const AUTO_SYNC_STALE_S = 300;      // pull on open if cache older than 5 min
const AUTO_SYNC_INTERVAL_MS = 300000; // and every 5 min while open

const HOUR_PX = 44;           // week/day row height
const LAYER_META = {
  augmentum: { label: 'Augmentum', dot: 'cal-dot-aug' },
  calendar:  { label: 'Calendar (devices)', dot: 'cal-dot-cal' },
  companion: { label: 'Companion', dot: 'cal-dot-comp' },
};
const COLORS = ['blue', 'green', 'amber', 'violet', 'rose', 'teal'];

function _esc(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/`/g, '&#96;').replace(/\$\{/g, '&#36;{');
}

// ── Date helpers (Monday-first week) ────────────────────────────────────

function _startOfDay(d) { const x = new Date(d); x.setHours(0, 0, 0, 0); return x; }
function _addDays(d, n) { const x = new Date(d); x.setDate(x.getDate() + n); return x; }
function _sameDay(a, b) {
  return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
}
function _startOfWeek(d) {
  const x = _startOfDay(d);
  const dow = (x.getDay() + 6) % 7;   // Mon=0 … Sun=6
  return _addDays(x, -dow);
}
function _monthLabel(d) {
  return d.toLocaleDateString(undefined, { month: 'long', year: 'numeric' });
}
function _fmtTime(d) {
  return d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
}
function _fmtDayHeading(d) {
  const today = new Date();
  if (_sameDay(d, today)) return `Today · ${d.toLocaleDateString(undefined, { weekday: 'long', month: 'short', day: 'numeric' })}`;
  return d.toLocaleDateString(undefined, { weekday: 'long', month: 'short', day: 'numeric' });
}
function _toLocalDateInput(d) {
  const x = new Date(d);
  return `${x.getFullYear()}-${String(x.getMonth() + 1).padStart(2, '0')}-${String(x.getDate()).padStart(2, '0')}`;
}
function _toLocalTimeInput(d) {
  const x = new Date(d);
  return `${String(x.getHours()).padStart(2, '0')}:${String(x.getMinutes()).padStart(2, '0')}`;
}
// Combine a YYYY-MM-DD + HH:MM (local) into a UTC ISO string for the API.
function _localToIso(dateStr, timeStr, allDay) {
  if (allDay) return dateStr;   // bare date for all-day
  const [y, m, d] = (dateStr || '').split('-').map((n) => parseInt(n, 10));
  const [hh, mm] = (timeStr || '00:00').split(':').map((n) => parseInt(n, 10));
  const dt = new Date(y, (m || 1) - 1, d || 1, hh || 0, mm || 0);
  return dt.toISOString();
}
// Parse an API instant ('...Z' or bare date) into a local Date.
function _parseInstant(s, allDay) {
  if (!s) return null;
  if (allDay || (typeof s === 'string' && s.length <= 10 && !s.includes('T'))) {
    const [y, m, d] = s.split('-').map((n) => parseInt(n, 10));
    return new Date(y, (m || 1) - 1, d || 1);
  }
  const t = Date.parse(s.includes('T') ? s : s.replace(' ', 'T') + 'Z');
  return Number.isFinite(t) ? new Date(t) : null;
}

// ── API ─────────────────────────────────────────────────────────────────

function _visibleRange() {
  if (_view === 'month') {
    const first = _startOfWeek(new Date(_cursor.getFullYear(), _cursor.getMonth(), 1));
    return { start: first, end: _addDays(first, 42) };
  }
  if (_view === 'week') {
    const s = _startOfWeek(_cursor);
    return { start: s, end: _addDays(s, 7) };
  }
  if (_view === 'day') {
    const s = _startOfDay(_cursor);
    return { start: s, end: _addDays(s, 1) };
  }
  // agenda: 60 days forward from the cursor day
  const s = _startOfDay(_cursor);
  return { start: s, end: _addDays(s, 60) };
}

async function _fetchEvents() {
  const { start, end } = _visibleRange();
  const layers = Object.keys(_layers).filter((k) => _layers[k]).join(',');
  const url = `/api/calendar/events?start=${encodeURIComponent(start.toISOString())}`
    + `&end=${encodeURIComponent(end.toISOString())}&layers=${encodeURIComponent(layers)}`;
  const seq = ++_fetchSeq;
  try {
    const resp = await fetch(url, { credentials: 'same-origin' });
    if (seq !== _fetchSeq) return null;   // superseded by a newer navigation
    if (!resp.ok) return [];
    const data = await resp.json();
    return Array.isArray(data.events) ? data.events : [];
  } catch (_) { return []; }
}

async function _fetchServices() {
  try {
    const resp = await fetch('/api/calendar/services', { credentials: 'same-origin' });
    if (!resp.ok) return;
    const data = await resp.json();
    _syncAvailable = !!data.sync_available;
    _lastSynced = Number(data.last_synced_at) || 0;
  } catch (_) { _syncAvailable = false; }
}

// Automatic CalDAV pull. force=true always syncs; otherwise the server's
// freshness gate (?if_stale_seconds) makes a recently-synced cache a cheap
// no-op, so calling this on open / on an interval is safe. Refreshes the
// grid only when events actually changed (a skipped no-op leaves it alone).
async function _autoSync(force) {
  if (!_syncAvailable || _syncing) return;
  _syncing = true;
  _updateSyncChip();
  try {
    const url = force ? '/api/calendar/sync' : `/api/calendar/sync?if_stale_seconds=${AUTO_SYNC_STALE_S}`;
    const resp = await fetch(url, { method: 'POST', credentials: 'same-origin' });
    const data = resp.ok ? await resp.json() : null;
    if (data && data.ok) {
      if (data.last_synced_at) _lastSynced = Number(data.last_synced_at);
      if (!data.skipped) { _syncing = false; await _refresh(); return; }
    }
  } catch (_) { /* offline — leave the cache as-is */ }
  _syncing = false;
  _updateSyncChip();
}

function _syncStatusText() {
  if (_syncing) return '🔄 Syncing…';
  if (!_lastSynced) return '🔄 Sync';
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - _lastSynced));
  if (secs < 60) return '🔄 Synced just now';
  if (secs < 3600) return `🔄 Synced ${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `🔄 Synced ${Math.floor(secs / 3600)}h ago`;
  return `🔄 Synced ${Math.floor(secs / 86400)}d ago`;
}

function _updateSyncChip() {
  const btn = _modal && _modal.querySelector('.cal-sync');
  if (btn) { btn.textContent = _syncStatusText(); btn.disabled = _syncing; }
}

async function _saveEvent(payload, id) {
  const url = id ? `/api/calendar/events/${id}` : '/api/calendar/events';
  const method = id ? 'PATCH' : 'POST';
  try {
    const resp = await fetch(url, {
      method, headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin', body: JSON.stringify(payload),
    });
    if (!resp.ok) return { ok: false };
    return await resp.json();
  } catch (_) { return { ok: false }; }
}

async function _deleteEvent(id) {
  try {
    const resp = await fetch(`/api/calendar/events/${id}`, {
      method: 'DELETE', credentials: 'same-origin',
    });
    return resp.ok;
  } catch (_) { return false; }
}

// Normalize a raw API event into an instance with local Date bounds.
function _decorate(ev) {
  const start = _parseInstant(ev.start, ev.all_day);
  let end = _parseInstant(ev.end, ev.all_day) || start;
  if (end && start && end < start) end = start;
  return { ...ev, _start: start, _end: end };
}

// ── Header ───────────────────────────────────────────────────────────────

function _rangeTitle() {
  if (_view === 'month') return _monthLabel(_cursor);
  if (_view === 'day') return _cursor.toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' });
  if (_view === 'week') {
    const s = _startOfWeek(_cursor); const e = _addDays(s, 6);
    const sameMonth = s.getMonth() === e.getMonth();
    const sM = s.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    const eM = e.toLocaleDateString(undefined, sameMonth ? { day: 'numeric', year: 'numeric' } : { month: 'short', day: 'numeric', year: 'numeric' });
    return `${sM} – ${eM}`;
  }
  return `From ${_cursor.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}`;
}

function _headerHtml() {
  const views = ['month', 'week', 'day', 'agenda'];
  const tabs = views.map((v) =>
    `<button type="button" class="cal-tab ${_view === v ? 'on' : ''}" data-view="${v}">${v[0].toUpperCase() + v.slice(1)}</button>`).join('');
  const legend = Object.keys(LAYER_META).map((k) =>
    `<button type="button" class="cal-legend ${_layers[k] ? 'on' : 'off'}" data-layer="${k}">
       <span class="cal-dot ${LAYER_META[k].dot}"></span>${_esc(LAYER_META[k].label)}</button>`).join('');
  // No calendar server yet → surface the one-tap way to connect one, so the
  // sync/device path is discoverable in the UI rather than a hidden feature.
  const connect = _syncAvailable ? '' :
    `<button type="button" class="cal-legend cal-connect" title="Install a calendar server (Radicale) to sync with your phone, laptop & tablet">＋ Connect a calendar</button>`;
  const sync = _syncAvailable
    ? `<button type="button" class="cal-btn cal-sync" title="Auto-syncs with your connected calendar. Click to refresh now.">${_esc(_syncStatusText())}</button>`
    : '';
  const devices = _syncAvailable
    ? `<button type="button" class="cal-btn cal-devices" title="Sync this calendar to your phone, tablet or laptop over your local network">📱 Devices</button>`
    : '';
  return `
    <header class="cal-header">
      <div class="cal-header-row">
        <span class="cal-title">Calendar</span>
        <div class="cal-tabs">${tabs}</div>
        <div class="cal-nav">
          <button type="button" class="cal-btn cal-nav-btn" data-nav="prev" aria-label="Previous">‹</button>
          <button type="button" class="cal-btn cal-today">Today</button>
          <button type="button" class="cal-btn cal-nav-btn" data-nav="next" aria-label="Next">›</button>
        </div>
        <span class="cal-range">${_esc(_rangeTitle())}</span>
        <div class="cal-header-actions">
          ${devices}
          ${sync}
          <button type="button" class="cal-btn cal-btn-primary cal-new">+ New</button>
        </div>
        <button type="button" class="cal-close" aria-label="Close">×</button>
      </div>
      <div class="cal-legend-row">${legend}${connect}</div>
    </header>`;
}

// ── Month view ────────────────────────────────────────────────────────────

function _renderMonth(body) {
  const dow = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const first = _startOfWeek(new Date(_cursor.getFullYear(), _cursor.getMonth(), 1));
  const today = new Date();
  const byDay = new Map();
  for (const ev of _events) {
    if (!ev._start) continue;
    const key = _toLocalDateInput(ev._start);
    (byDay.get(key) || byDay.set(key, []).get(key)).push(ev);
  }
  let cells = '';
  for (let i = 0; i < 42; i++) {
    const day = _addDays(first, i);
    const key = _toLocalDateInput(day);
    const inMonth = day.getMonth() === _cursor.getMonth();
    const isToday = _sameDay(day, today);
    const items = (byDay.get(key) || []).sort((a, b) => a._start - b._start);
    const shown = items.slice(0, 3).map((ev) => {
      const time = ev.all_day ? '' : `${_fmtTime(ev._start)} `;
      return `<button type="button" class="cal-chip cal-c-${_esc(ev.color)}" data-id="${_esc(ev.id)}" title="${_esc(ev.title)}">
        ${ev.opens_task ? '⟳ ' : ''}<span class="cal-chip-t">${_esc(time)}</span>${_esc(ev.title)}</button>`;
    }).join('');
    const more = items.length > 3 ? `<button type="button" class="cal-more" data-day="${key}">+${items.length - 3} more</button>` : '';
    cells += `<div class="cal-cell ${inMonth ? '' : 'cal-cell-dim'} ${isToday ? 'cal-cell-today' : ''}" data-day="${key}">
        <div class="cal-cell-num">${day.getDate()}</div>
        <div class="cal-cell-events">${shown}${more}</div>
      </div>`;
  }
  body.innerHTML = `
    <div class="cal-month">
      <div class="cal-month-dow">${dow.map((d) => `<div>${d}</div>`).join('')}</div>
      <div class="cal-month-grid">${cells}</div>
    </div>`;
  body.querySelectorAll('.cal-cell').forEach((cell) => {
    cell.addEventListener('click', (e) => {
      if (e.target.closest('.cal-chip') || e.target.closest('.cal-more')) return;
      const [y, m, d] = cell.dataset.day.split('-').map((n) => parseInt(n, 10));
      _openSheet(null, new Date(y, m - 1, d, 9, 0));
    });
  });
  _wireChips(body);
  body.querySelectorAll('.cal-more').forEach((b) =>
    b.addEventListener('click', () => { _view = 'day'; const [y, m, d] = b.dataset.day.split('-').map((n) => parseInt(n, 10)); _cursor = new Date(y, m - 1, d); _refresh(); }));
}

// ── Week / Day time-grid ──────────────────────────────────────────────────

function _renderTimeGrid(body, days) {
  const today = new Date();
  const hours = Array.from({ length: 24 }, (_, h) => h);
  const timeCol = hours.map((h) => {
    const label = h === 0 ? '' : new Date(2000, 0, 1, h).toLocaleTimeString(undefined, { hour: 'numeric' });
    return `<div class="cal-hour" style="height:${HOUR_PX}px">${label}</div>`;
  }).join('');

  const dayCols = days.map((day) => {
    const dayEvents = _events.filter((ev) => ev._start && !ev.all_day && _sameDay(ev._start, day));
    const blocks = _layoutDay(dayEvents).map(({ ev, col, cols }) => {
      const startMin = ev._start.getHours() * 60 + ev._start.getMinutes();
      const endMin = Math.max(startMin + 20, ev._end ? (ev._end.getHours() * 60 + ev._end.getMinutes()) || (startMin + 30) : startMin + 30);
      const top = (startMin / 60) * HOUR_PX;
      const height = Math.max(18, ((endMin - startMin) / 60) * HOUR_PX - 2);
      const width = 100 / cols;
      return `<button type="button" class="cal-block cal-c-${_esc(ev.color)}" data-id="${_esc(ev.id)}"
          style="top:${top}px;height:${height}px;left:${col * width}%;width:calc(${width}% - 3px)" title="${_esc(ev.title)}">
          <span class="cal-block-t">${_fmtTime(ev._start)}</span> ${ev.opens_task ? '⟳ ' : ''}${_esc(ev.title)}</button>`;
    }).join('');
    const allday = _events.filter((ev) => ev._start && ev.all_day && _sameDay(ev._start, day))
      .map((ev) => `<button type="button" class="cal-chip cal-c-${_esc(ev.color)}" data-id="${_esc(ev.id)}">${_esc(ev.title)}</button>`).join('');
    return `<div class="cal-daycol" data-day="${_toLocalDateInput(day)}">
        <div class="cal-daycol-allday">${allday}</div>
        <div class="cal-daycol-body" style="height:${HOUR_PX * 24}px">
          ${hours.map((h) => `<div class="cal-slot" data-hour="${h}" style="height:${HOUR_PX}px"></div>`).join('')}
          ${blocks}
        </div>
      </div>`;
  }).join('');

  const heads = days.map((day) => {
    const isToday = _sameDay(day, today);
    return `<div class="cal-dayhead ${isToday ? 'cal-dayhead-today' : ''}">
        <span class="cal-dayhead-dow">${day.toLocaleDateString(undefined, { weekday: 'short' })}</span>
        <span class="cal-dayhead-num">${day.getDate()}</span></div>`;
  }).join('');

  body.innerHTML = `
    <div class="cal-timegrid">
      <div class="cal-timegrid-head"><div class="cal-timegutter-head"></div>${heads}</div>
      <div class="cal-timegrid-scroll">
        <div class="cal-timegrid-body">
          <div class="cal-timegutter">${timeCol}</div>
          <div class="cal-timegrid-cols" style="grid-template-columns:repeat(${days.length},1fr)">${dayCols}</div>
        </div>
      </div>
    </div>`;
  // Scroll to 7am on open.
  const scroller = body.querySelector('.cal-timegrid-scroll');
  if (scroller) scroller.scrollTop = 7 * HOUR_PX;
  _wireChips(body);
  body.querySelectorAll('.cal-slot').forEach((slot) => {
    slot.addEventListener('click', () => {
      const col = slot.closest('.cal-daycol');
      const [y, m, d] = col.dataset.day.split('-').map((n) => parseInt(n, 10));
      _openSheet(null, new Date(y, m - 1, d, parseInt(slot.dataset.hour, 10), 0));
    });
  });
}

// Simple greedy column layout for overlapping events in one day.
function _layoutDay(evts) {
  const sorted = [...evts].sort((a, b) => a._start - b._start);
  const columns = [];   // each column = list of placed events
  const placed = [];
  for (const ev of sorted) {
    let ci = 0;
    for (; ci < columns.length; ci++) {
      const last = columns[ci][columns[ci].length - 1];
      if (last._end <= ev._start) break;
    }
    if (!columns[ci]) columns[ci] = [];
    columns[ci].push(ev);
    placed.push({ ev, col: ci });
  }
  const total = Math.max(1, columns.length);
  return placed.map((p) => ({ ...p, cols: total }));
}

// ── Agenda view ───────────────────────────────────────────────────────────

function _renderAgenda(body) {
  const groups = new Map();
  for (const ev of _events) {
    if (!ev._start) continue;
    const key = _toLocalDateInput(ev._start);
    (groups.get(key) || groups.set(key, []).get(key)).push(ev);
  }
  const keys = [...groups.keys()].sort();
  if (!keys.length) {
    body.innerHTML = `<div class="cal-agenda"><div class="cal-empty">
        <p>Nothing scheduled in this window.</p>
        <button type="button" class="cal-btn cal-btn-primary cal-new">+ New event</button></div></div>`;
    body.querySelector('.cal-new')?.addEventListener('click', () => _openSheet(null, new Date(_cursor)));
    return;
  }
  const html = keys.map((key) => {
    const [y, m, d] = key.split('-').map((n) => parseInt(n, 10));
    const day = new Date(y, m - 1, d);
    const rows = groups.get(key).sort((a, b) => a._start - b._start).map((ev) => {
      const time = ev.all_day ? 'All day' : _fmtTime(ev._start);
      const loc = ev.location ? `<span class="cal-ag-loc">📍 ${_esc(ev.location)}</span>` : '';
      const synced = ev.synced ? '<span class="cal-ag-badge">🔄 device</span>' : '';
      return `<button type="button" class="cal-ag-row" data-id="${_esc(ev.id)}">
          <span class="cal-ag-time">${_esc(time)}</span>
          <span class="cal-dot ${LAYER_META[ev.layer].dot}"></span>
          <span class="cal-ag-title">${ev.opens_task ? '⟳ ' : ''}${_esc(ev.title)}</span>
          <span class="cal-ag-layer">${_esc(LAYER_META[ev.layer].label)}</span>
          ${loc}${synced}
        </button>`;
    }).join('');
    return `<div class="cal-ag-group">
        <div class="cal-ag-heading">${_esc(_fmtDayHeading(day))}</div>${rows}</div>`;
  }).join('');
  body.innerHTML = `<div class="cal-agenda">${html}</div>`;
  _wireChips(body);
}

// ── Chip / block / row wiring ─────────────────────────────────────────────

function _wireChips(scope) {
  scope.querySelectorAll('[data-id]').forEach((el) => {
    el.addEventListener('click', (e) => {
      e.stopPropagation();
      const ev = _events.find((x) => x.id === el.dataset.id);
      if (!ev) return;
      if (ev.opens_task && ev.task_id != null) {
        import('../companion-topics-modal.js').then((m) => m.openTask?.(ev.task_id)).catch(() => {});
        return;
      }
      if (ev.layer === 'augmentum') _openSheet(ev);
      else _openReadOnly(ev);
    });
  });
}

// ── Body dispatch ─────────────────────────────────────────────────────────

function _renderBody() {
  const body = _modal.querySelector('.cal-body');
  if (!body) return;
  if (_view === 'month') _renderMonth(body);
  else if (_view === 'week') _renderTimeGrid(body, Array.from({ length: 7 }, (_, i) => _addDays(_startOfWeek(_cursor), i)));
  else if (_view === 'day') _renderTimeGrid(body, [_startOfDay(_cursor)]);
  else _renderAgenda(body);
}

function _renderHeader() {
  const head = _modal.querySelector('.cal-header');
  if (head) head.outerHTML = _headerHtml();
  _wireHeader();
}

function _wireHeader() {
  const q = (s) => _modal.querySelector(s);
  q('.cal-close')?.addEventListener('click', close);
  q('.cal-new')?.addEventListener('click', () => _openSheet(null, new Date(_cursor)));
  q('.cal-today')?.addEventListener('click', () => { _cursor = new Date(); _refresh(); });
  q('.cal-sync')?.addEventListener('click', () => _autoSync(true));
  _modal.querySelectorAll('.cal-tab').forEach((b) =>
    b.addEventListener('click', () => { _view = b.dataset.view; _refresh(); }));
  _modal.querySelectorAll('.cal-nav-btn').forEach((b) =>
    b.addEventListener('click', () => _navigate(b.dataset.nav === 'next' ? 1 : -1)));
  _modal.querySelectorAll('.cal-legend[data-layer]').forEach((b) =>
    b.addEventListener('click', () => { const k = b.dataset.layer; _layers[k] = !_layers[k]; _refresh(); }));
  q('.cal-devices')?.addEventListener('click', _openDevicesCard);
  q('.cal-connect')?.addEventListener('click', () => {
    // Open Discover to where the calendar server (Radicale) lives so the user
    // can install it one-tap; it auto-detects here once running.
    close();
    import('../discover/index.js').then((m) => m.openDiscover?.({ search: 'calendar' })).catch(() => {});
  });
}

function _navigate(dir) {
  if (_view === 'month') _cursor = new Date(_cursor.getFullYear(), _cursor.getMonth() + dir, 1);
  else if (_view === 'week') _cursor = _addDays(_cursor, 7 * dir);
  else if (_view === 'day') _cursor = _addDays(_cursor, dir);
  else _cursor = _addDays(_cursor, 30 * dir);
  _refresh();
}

// ── Create / edit sheet ───────────────────────────────────────────────────

function _openSheet(ev, when) {
  const editing = ev && ev.layer === 'augmentum' ? ev : null;
  const start = editing ? editing._start : (when || new Date());
  const end = editing && editing._end ? editing._end : new Date(start.getTime() + 30 * 60000);
  const allDay = editing ? !!editing.all_day : false;
  const color = editing ? (editing.color || 'blue') : 'blue';

  const swatches = COLORS.map((c) =>
    `<button type="button" class="cal-swatch cal-c-${c} ${c === color ? 'on' : ''}" data-color="${c}" aria-label="${c}"></button>`).join('');
  const repeatOpts = [
    ['', 'Does not repeat'], ['FREQ=DAILY', 'Every day'],
    ['FREQ=WEEKLY', 'Every week'], ['FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR', 'Every weekday'],
    ['FREQ=MONTHLY', 'Every month'], ['FREQ=YEARLY', 'Every year'],
  ];
  const curRule = editing ? (editing.rrule || '') : '';
  const repeatSel = repeatOpts.map(([v, l]) =>
    `<option value="${_esc(v)}" ${v === curRule ? 'selected' : ''}>${_esc(l)}</option>`).join('');

  const syncRow = _syncAvailable ? `
      <label class="cal-sheet-sync">
        <input type="checkbox" class="cal-f-sync" ${editing && editing.synced ? 'checked disabled' : ''}>
        <span><strong>🔄 Also add to my devices</strong><br><span class="cal-hint">Syncs to your connected calendar over open standards (CalDAV).</span></span>
      </label>` : '';

  const sheet = document.createElement('div');
  sheet.className = 'cal-sheet-wrap';
  sheet.innerHTML = `
    <div class="cal-sheet-backdrop"></div>
    <div class="cal-sheet" role="dialog" aria-label="${editing ? 'Edit event' : 'New event'}">
      <div class="cal-sheet-head">
        <span class="cal-sheet-title">${editing ? 'Edit event' : 'New event'}</span>
        <button type="button" class="cal-sheet-x" aria-label="Close">×</button>
      </div>
      <div class="cal-sheet-body">
        <input type="text" class="cal-input cal-f-title" placeholder="Add a title" value="${_esc(editing ? editing.title : '')}">
        <label class="cal-check"><input type="checkbox" class="cal-f-allday" ${allDay ? 'checked' : ''}> All day</label>
        <div class="cal-field-row">
          <label>Starts</label>
          <input type="date" class="cal-input cal-f-sdate" value="${_toLocalDateInput(start)}">
          <input type="time" class="cal-input cal-f-stime" value="${_toLocalTimeInput(start)}">
        </div>
        <div class="cal-field-row">
          <label>Ends</label>
          <input type="date" class="cal-input cal-f-edate" value="${_toLocalDateInput(end)}">
          <input type="time" class="cal-input cal-f-etime" value="${_toLocalTimeInput(end)}">
        </div>
        <div class="cal-field-row">
          <label>Repeat</label>
          <select class="cal-input cal-f-repeat">${repeatSel}</select>
        </div>
        <div class="cal-field-row">
          <label>Location</label>
          <input type="text" class="cal-input cal-f-loc" placeholder="Optional" value="${_esc(editing ? editing.location : '')}">
        </div>
        <textarea class="cal-input cal-f-notes" rows="2" placeholder="Notes (optional)">${_esc(editing ? editing.description : '')}</textarea>
        <div class="cal-field-row"><label>Color</label><div class="cal-swatches">${swatches}</div></div>
        ${syncRow}
        <div class="cal-sheet-error" hidden></div>
      </div>
      <div class="cal-sheet-foot">
        ${editing ? '<button type="button" class="cal-btn cal-btn-danger cal-f-delete">Delete</button>' : '<span></span>'}
        <div>
          <button type="button" class="cal-btn cal-f-cancel">Cancel</button>
          <button type="button" class="cal-btn cal-btn-primary cal-f-save">${editing ? 'Save' : 'Create'}</button>
        </div>
      </div>
    </div>`;
  _modal.appendChild(sheet);

  const close = () => sheet.remove();
  sheet.querySelector('.cal-sheet-backdrop').addEventListener('click', close);
  sheet.querySelector('.cal-sheet-x').addEventListener('click', close);
  sheet.querySelector('.cal-f-cancel').addEventListener('click', close);
  sheet.querySelectorAll('.cal-swatch').forEach((s) =>
    s.addEventListener('click', () => {
      sheet.querySelectorAll('.cal-swatch').forEach((x) => x.classList.remove('on'));
      s.classList.add('on');
    }));
  const allDayBox = sheet.querySelector('.cal-f-allday');
  const timeInputs = () => sheet.querySelectorAll('.cal-f-stime, .cal-f-etime');
  const applyAllDay = () => timeInputs().forEach((t) => { t.style.display = allDayBox.checked ? 'none' : ''; });
  allDayBox.addEventListener('change', applyAllDay); applyAllDay();
  sheet.querySelector('.cal-f-title').focus();

  sheet.querySelector('.cal-f-save').addEventListener('click', async () => {
    const errEl = sheet.querySelector('.cal-sheet-error');
    const title = sheet.querySelector('.cal-f-title').value.trim();
    if (!title) { errEl.textContent = 'Give it a title.'; errEl.hidden = false; return; }
    const isAllDay = allDayBox.checked;
    const payload = {
      title,
      start: _localToIso(sheet.querySelector('.cal-f-sdate').value, sheet.querySelector('.cal-f-stime').value, isAllDay),
      end: _localToIso(sheet.querySelector('.cal-f-edate').value, sheet.querySelector('.cal-f-etime').value, isAllDay),
      all_day: isAllDay,
      location: sheet.querySelector('.cal-f-loc').value.trim(),
      description: sheet.querySelector('.cal-f-notes').value.trim(),
      color: sheet.querySelector('.cal-swatch.on')?.dataset.color || 'blue',
      rrule: sheet.querySelector('.cal-f-repeat').value,
    };
    const syncBox = sheet.querySelector('.cal-f-sync');
    if (syncBox && syncBox.checked && !syncBox.disabled) payload.sync_to_devices = true;
    const btn = sheet.querySelector('.cal-f-save'); btn.disabled = true; btn.textContent = 'Saving…';
    const res = await _saveEvent(payload, editing ? editing.native_id : null);
    if (res && res.ok) { close(); _refresh(); }
    else { btn.disabled = false; btn.textContent = editing ? 'Save' : 'Create'; errEl.textContent = 'Could not save. Try again.'; errEl.hidden = false; }
  });

  sheet.querySelector('.cal-f-delete')?.addEventListener('click', async () => {
    if (!window.confirm('Delete this event?')) return;
    if (await _deleteEvent(editing.native_id)) { close(); _refresh(); }
  });
}

// Per-platform CalDAV setup instructions. Adding a CalDAV account is a manual
// OS step on every platform — so the exact tap-path lives right here in the
// flow, next to the copy-paste fields, rather than in external docs.
const DEVICE_GUIDES = [
  { id: 'ios', label: 'iPhone / iPad', steps: [
    'Open <strong>Settings</strong> → <strong>Calendar</strong> → <strong>Accounts</strong> → <strong>Add Account</strong>.',
    'Choose <strong>Other</strong> → <strong>Add CalDAV Account</strong>.',
    'Server = the <strong>Server URL</strong> below · User Name / Password = below. Leave Description as you like.',
    'Tap <strong>Next</strong> → <strong>Save</strong>. Your events appear in the Calendar app.',
  ] },
  { id: 'android', label: 'Android', steps: [
    'Install <strong>DAVx5</strong> (free, from the Play Store or F-Droid) — Android has no built-in CalDAV.',
    'Open DAVx5 → <strong>+</strong> → <strong>Login with URL and user name</strong>.',
    'Base URL = the <strong>Server URL</strong> below · enter the User Name & Password below.',
    'Tap <strong>Login</strong>, pick the calendar, and let it sync. It shows in Google Calendar / your calendar app.',
  ] },
  { id: 'macos', label: 'Mac', steps: [
    'Open <strong>Calendar</strong> → menu <strong>Calendar</strong> → <strong>Add Account…</strong> → <strong>Other CalDAV Account</strong>.',
    'Account Type = <strong>Manual</strong>.',
    'User Name / Password = below · Server Address = the <strong>Server URL</strong> below.',
    'Click <strong>Sign In</strong>. Events sync into the Calendar app.',
  ] },
  { id: 'thunderbird', label: 'Thunderbird', steps: [
    'Open <strong>Calendar</strong> → right-click the calendar list → <strong>New Calendar</strong>.',
    'Choose <strong>On the Network</strong> → <strong>CalDAV</strong>.',
    'Location = the <strong>Server URL</strong> below · Username = below (you\'ll be prompted for the password).',
    'Click <strong>Find Calendars</strong> → <strong>Subscribe</strong>.',
  ] },
  { id: 'other', label: 'Other', steps: [
    'Any CalDAV client works (GNOME Calendar, eM Client, etc.).',
    'Add a <strong>CalDAV</strong> / <strong>network calendar</strong> account.',
    'Use the <strong>Server URL</strong>, <strong>Username</strong>, and <strong>Password</strong> below.',
  ] },
];

// "Add to your devices" — copy-paste CalDAV setup for phones/laptops on the
// local network, with the full per-platform instructions inline. Fetches the
// connection details on open.
async function _openDevicesCard() {
  let info = null;
  try {
    const resp = await fetch('/api/calendar/connection', { credentials: 'same-origin' });
    if (resp.ok) info = await resp.json();
  } catch (_) { /* offline */ }

  const sheet = document.createElement('div');
  sheet.className = 'cal-sheet-wrap';
  const field = (label, value, mono) => `
    <div class="cal-dev-field">
      <label>${_esc(label)}</label>
      <div class="cal-dev-copy">
        <input type="text" class="cal-input ${mono ? 'cal-mono' : ''}" readonly value="${_esc(value)}">
        <button type="button" class="cal-btn cal-copy" data-copy="${_esc(value)}">Copy</button>
      </div>
    </div>`;

  let body;
  if (!info || !info.installed) {
    body = `<div class="cal-hint">No calendar server is connected yet. Use
      <strong>＋ Connect a calendar</strong> first, then come back here to set up your devices.</div>`;
  } else if (!info.lan_published) {
    body = `
      <div class="cal-hint">Your calendar server is running, but isn't published on your
      local network yet, so phones and laptops can't reach it. Once it's LAN-published,
      the exact address and login will appear here with step-by-step device instructions.</div>
      ${field('Calendar path', info.path, true)}
      ${info.username ? field('Username', info.username) : ''}
      ${info.password ? field('Password', info.password, true) : ''}`;
  } else {
    const tabs = DEVICE_GUIDES.map((g, i) =>
      `<button type="button" class="cal-dev-tab ${i === 0 ? 'on' : ''}" data-guide="${g.id}">${_esc(g.label)}</button>`).join('');
    const panels = DEVICE_GUIDES.map((g, i) =>
      `<ol class="cal-dev-steps ${i === 0 ? '' : 'hidden'}" data-guide="${g.id}">
         ${g.steps.map((s) => `<li>${s}</li>`).join('')}
       </ol>`).join('');
    body = `
      <div class="cal-dev-note">📶 Make sure your device is on the <strong>same Wi-Fi / network</strong> as this machine.</div>
      <div class="cal-dev-tabs">${tabs}</div>
      <div class="cal-dev-panels">${panels}</div>
      <div class="cal-dev-creds">
        ${field('Server URL', info.lan_url, true)}
        ${field('Username', info.username)}
        ${field('Password', info.password, true)}
      </div>`;
  }

  sheet.innerHTML = `
    <div class="cal-sheet-backdrop"></div>
    <div class="cal-sheet cal-sheet-wide" role="dialog" aria-label="Add to your devices">
      <div class="cal-sheet-head">
        <span class="cal-sheet-title">📱 Add to your devices</span>
        <button type="button" class="cal-sheet-x" aria-label="Close">×</button>
      </div>
      <div class="cal-sheet-body">${body}</div>
      <div class="cal-sheet-foot"><span></span><div>
        <button type="button" class="cal-btn cal-btn-primary cal-f-cancel">Done</button>
      </div></div>
    </div>`;
  _modal.appendChild(sheet);
  const close = () => sheet.remove();
  sheet.querySelector('.cal-sheet-backdrop').addEventListener('click', close);
  sheet.querySelector('.cal-sheet-x').addEventListener('click', close);
  sheet.querySelector('.cal-f-cancel').addEventListener('click', close);
  // Platform tab switching.
  sheet.querySelectorAll('.cal-dev-tab').forEach((tab) =>
    tab.addEventListener('click', () => {
      sheet.querySelectorAll('.cal-dev-tab').forEach((t) => t.classList.toggle('on', t === tab));
      sheet.querySelectorAll('.cal-dev-steps').forEach((p) =>
        p.classList.toggle('hidden', p.dataset.guide !== tab.dataset.guide));
    }));
  sheet.querySelectorAll('.cal-copy').forEach((b) =>
    b.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(b.dataset.copy); b.textContent = 'Copied'; }
      catch (_) { b.textContent = 'Copy failed'; }
      setTimeout(() => { b.textContent = 'Copy'; }, 1400);
    }));
}

// Read-only detail for CalDAV events (edited on the source, not here).
function _openReadOnly(ev) {
  const sheet = document.createElement('div');
  sheet.className = 'cal-sheet-wrap';
  const when = ev.all_day ? _fmtDayHeading(ev._start)
    : `${_fmtDayHeading(ev._start)} · ${_fmtTime(ev._start)}${ev._end && ev._end > ev._start ? '–' + _fmtTime(ev._end) : ''}`;
  sheet.innerHTML = `
    <div class="cal-sheet-backdrop"></div>
    <div class="cal-sheet cal-sheet-ro" role="dialog" aria-label="Event details">
      <div class="cal-sheet-head">
        <span class="cal-sheet-title"><span class="cal-dot ${LAYER_META[ev.layer].dot}"></span> ${_esc(ev.title)}</span>
        <button type="button" class="cal-sheet-x" aria-label="Close">×</button>
      </div>
      <div class="cal-sheet-body">
        <div class="cal-ro-when">${_esc(when)}</div>
        ${ev.location ? `<div class="cal-ro-line">📍 ${_esc(ev.location)}</div>` : ''}
        ${ev.calendar_name ? `<div class="cal-ro-line">🗓 ${_esc(ev.calendar_name)}</div>` : ''}
        ${ev.description ? `<div class="cal-ro-notes">${_esc(ev.description)}</div>` : ''}
        <div class="cal-hint">Synced from your connected calendar — edit it on the source device.</div>
      </div>
    </div>`;
  _modal.appendChild(sheet);
  const close = () => sheet.remove();
  sheet.querySelector('.cal-sheet-backdrop').addEventListener('click', close);
  sheet.querySelector('.cal-sheet-x').addEventListener('click', close);
}

// ── Refresh / shell ────────────────────────────────────────────────────────

async function _refresh() {
  _renderHeader();
  const body = _modal.querySelector('.cal-body');
  if (body) body.innerHTML = `<div class="cal-loading">Loading…</div>`;
  const raw = await _fetchEvents();
  if (raw === null) return;   // superseded
  _events = raw.map(_decorate);
  _renderBody();
}

function _buildModal() {
  if (_modal) return _modal;
  const root = document.createElement('div');
  root.className = 'cal-modal hidden';
  root.innerHTML = `
    <div class="cal-backdrop"></div>
    <div class="cal-panel" role="dialog" aria-label="Calendar">
      ${_headerHtml()}
      <div class="cal-body"><div class="cal-loading">Loading…</div></div>
    </div>`;
  document.body.appendChild(root);
  _modal = root;
  root.querySelector('.cal-backdrop').addEventListener('click', close);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && _modal && !_modal.classList.contains('hidden')) {
      const sheet = _modal.querySelector('.cal-sheet-wrap');
      if (sheet) sheet.remove(); else close();
    }
  });
  return root;
}

export async function open() {
  _cursor = _cursor || new Date();
  _buildModal();
  _modal.classList.remove('hidden');
  await _fetchServices();
  _wireHeader();
  if (_dialog) { try { _dialog.release(); } catch (_) {} _dialog = null; }
  _dialog = installDialog(_modal.querySelector('.cal-panel'), {
    escapeCloses: false, initialFocus: '.cal-new', setAria: true,
  });
  await _refresh();
  // Automatic syncing where available: pull on open if the cache is stale,
  // then keep it fresh on an interval for as long as the calendar is open.
  if (_syncAvailable) {
    _autoSync(false);
    if (_syncTimer) clearInterval(_syncTimer);
    _syncTimer = setInterval(() => {
      if (_modal && !_modal.classList.contains('hidden')) _autoSync(false);
    }, AUTO_SYNC_INTERVAL_MS);
  }
}

export function close() {
  if (_dialog) { try { _dialog.release(); } catch (_) {} _dialog = null; }
  if (_syncTimer) { clearInterval(_syncTimer); _syncTimer = null; }
  if (_modal) {
    _modal.querySelectorAll('.cal-sheet-wrap').forEach((s) => s.remove());
    _modal.classList.add('hidden');
  }
}
