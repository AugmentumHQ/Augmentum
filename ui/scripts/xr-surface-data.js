/*
 * XR Surface Data
 *
 * Lightweight snapshot collector for headset-native panels. It keeps the XR
 * scene stereo-native while feeding panels with actual Augmentum state from
 * the live DOM, app events, and small same-origin API reads.
 */

const DEFAULT_POLL_MS = 6500;
const MAX_ITEMS = 6;
const MAX_VOICE_MESSAGES = 12;

function _now() {
  return Date.now();
}

function _clean(value, max = 180) {
  const text = String(value ?? '').replace(/\s+/g, ' ').trim();
  return max > 0 && text.length > max ? `${text.slice(0, Math.max(0, max - 3))}...` : text;
}

function _num(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function _dateish(value) {
  const text = _clean(value, 40);
  if (!text) return '';
  const t = new Date(text).getTime();
  if (!Number.isFinite(t)) return text.slice(0, 10);
  const diff = Date.now() - t;
  if (diff < 60_000) return 'just now';
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return new Date(t).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function _item(label, detail = '', extra = {}) {
  return {
    label: _clean(label || 'Untitled', 90),
    detail: _clean(detail, 130),
    ...extra,
  };
}

function _signature(payload) {
  try {
    return JSON.stringify({
      title: payload.title || '',
      summary: payload.summary || '',
      status: payload.status || '',
      metrics: payload.metrics || [],
      lines: payload.lines || [],
      items: payload.items || [],
    });
  } catch {
    return String(Math.random());
  }
}

async function _fetchJson(url) {
  const resp = await fetch(url, { cache: 'no-store' });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

function _firstArray(data, keys) {
  if (Array.isArray(data)) return data;
  for (const key of keys) {
    if (Array.isArray(data?.[key])) return data[key];
  }
  return [];
}

function _sourceMeta(entry) {
  const meta = entry?.source_metadata;
  return meta && typeof meta === 'object' ? meta : {};
}

function _fileKind(entry) {
  return _clean(entry?.kind || entry?.mime_type || entry?.source || 'file', 30);
}

function _fileProgress(entry) {
  const meta = _sourceMeta(entry);
  const pct = _num(meta.progress_pct ?? meta.progress ?? entry?.progress_pct, 0);
  if (pct <= 0) return '';
  const normalized = pct <= 1 ? pct * 100 : pct;
  return `${Math.round(Math.max(0, Math.min(100, normalized)))}%`;
}

function _fileDetail(entry) {
  const parts = [];
  const kind = _fileKind(entry);
  if (kind) parts.push(kind);
  const progress = _fileProgress(entry);
  if (progress) parts.push(progress);
  const updated = _dateish(entry?.updated_at || entry?.created_at);
  if (updated) parts.push(updated);
  return parts.join(' | ');
}

function _artifactDetail(entry) {
  const parts = [];
  if (entry?.format) parts.push(String(entry.format).toUpperCase());
  if (entry?.metadata?.kind) parts.push(_clean(entry.metadata.kind, 30));
  const when = _dateish(entry?.last_opened_at || entry?.created_at);
  if (when) parts.push(when);
  return parts.join(' | ');
}

function _queryText(selector, root = document) {
  return _clean(root?.querySelector?.(selector)?.textContent || '', 160);
}

function _visibleText(selector, maxItems = MAX_ITEMS) {
  return Array.from(document.querySelectorAll(selector))
    .filter((el) => {
      const rect = el.getBoundingClientRect?.();
      return !rect || rect.width > 0 || rect.height > 0;
    })
    .map((el) => _clean(el.textContent, 160))
    .filter(Boolean)
    .slice(0, maxItems);
}

class XrSurfaceDataStore {
  constructor({ pollMs = DEFAULT_POLL_MS } = {}) {
    this.pollMs = Math.max(2500, Number(pollMs) || DEFAULT_POLL_MS);
    this.snapshots = new Map();
    this.signatures = new Map();
    this.voiceMessages = [];
    this.agenticSnapshot = null;
    this.narrativeSnapshot = null;
    this.browseSnapshot = null;
    this.mediaState = null;
    this.versionCounter = 1;
    this.timer = null;
    this.started = false;
    this.listeners = [];
    this.inflight = false;
  }

  start() {
    if (this.started) return this;
    this.started = true;
    this._wireEvents();
    this.refreshAll();
    this.timer = setInterval(() => this.refreshAll(), this.pollMs);
    return this;
  }

  stop() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    for (const [target, name, fn] of this.listeners) {
      try { target.removeEventListener(name, fn); } catch {}
    }
    this.listeners = [];
    this.started = false;
  }

  snapshot(action) {
    return this.snapshots.get(String(action || '').trim()) || null;
  }

  version(action) {
    return this.snapshot(action)?.version || 0;
  }

  refreshAll() {
    this._refreshDomSurfaces();
    if (this.inflight) return;
    this.inflight = true;
    Promise.allSettled([
      this._refreshFiles(),
      this._refreshNotes(),
      this._refreshStudio(),
      this._refreshMedia(),
      this._refreshDevices(),
      this._refreshGames(),
    ]).finally(() => {
      this.inflight = false;
      this._refreshDomSurfaces();
    });
  }

  _listen(target, name, fn) {
    if (!target?.addEventListener) return;
    target.addEventListener(name, fn);
    this.listeners.push([target, name, fn]);
  }

  _wireEvents() {
    const doc = typeof document !== 'undefined' ? document : null;
    const win = typeof window !== 'undefined' ? window : null;
    this._listen(doc, 'augmentum:voice-message', (event) => {
      const detail = event.detail || {};
      const role = _clean(detail.role || 'voice', 24);
      const text = _clean(detail.text || '', 260);
      if (!text) return;
      this.voiceMessages.push({ role, text, at: _now(), sessionId: detail.sessionId || '' });
      this.voiceMessages.splice(0, Math.max(0, this.voiceMessages.length - MAX_VOICE_MESSAGES));
      this._refreshChat();
    });
    this._listen(doc, 'augmentum:agentic-task-snapshot', (event) => {
      this.agenticSnapshot = event.detail || null;
      this._refreshAgentic();
    });
    this._listen(doc, 'augmentum:narrative-state', (event) => {
      this.narrativeSnapshot = event.detail || null;
      this._refreshNarrative();
    });
    this._listen(doc, 'augmentum:browse-extracted', (event) => {
      this.browseSnapshot = event.detail || null;
      this._refreshBrowse();
    });
    this._listen(win, 'media-player:progress', (event) => {
      this.mediaState = event.detail || null;
      this._refreshMediaFromState();
    });
    this._listen(win, 'floating-video:state', (event) => {
      this.mediaState = event.detail || this.mediaState;
      this._refreshMediaFromState();
    });
    ['augmentum:sessions-rendered', 'augmentum:session-changed', 'augmentum:mode-changed']
      .forEach((name) => this._listen(doc, name, () => this._refreshDomSurfaces()));
    ['media:queue-updated', 'media-servers:changed', 'library:games-source-refresh']
      .forEach((name) => this._listen(win, name, () => this.refreshAll()));
    ['artifact:saved', 'augmentum:image-generated', 'augmentum:bookmarks-changed']
      .forEach((name) => this._listen(doc, name, () => this.refreshAll()));
  }

  _set(action, payload) {
    const key = String(action || '').trim();
    if (!key) return;
    const normalized = {
      action: key,
      title: _clean(payload.title || key, 80),
      summary: _clean(payload.summary || '', 220),
      status: _clean(payload.status || '', 80),
      metrics: Array.isArray(payload.metrics) ? payload.metrics.map((m) => _clean(m, 60)).filter(Boolean) : [],
      lines: Array.isArray(payload.lines) ? payload.lines.map((l) => _clean(l, 150)).filter(Boolean) : [],
      items: Array.isArray(payload.items) ? payload.items.filter((i) => i?.label).slice(0, MAX_ITEMS) : [],
      updatedAt: _now(),
      source: payload.source || 'live',
    };
    const sig = _signature(normalized);
    const priorSig = this.signatures.get(key);
    const prior = this.snapshots.get(key);
    if (sig === priorSig && prior) {
      prior.updatedAt = normalized.updatedAt;
      return;
    }
    normalized.version = this.versionCounter++;
    this.signatures.set(key, sig);
    this.snapshots.set(key, normalized);
  }

  _refreshDomSurfaces() {
    this._refreshChat();
    this._refreshAgentic();
    this._refreshAnalytical();
    this._refreshNarrative();
    this._refreshCoder();
    this._refreshBrowse();
    this._refreshMediaFromState();
  }

  _refreshChat() {
    const domMessages = Array.from(document.querySelectorAll('#chat-messages .message'))
      .slice(-5)
      .map((el) => {
        const role = el.classList.contains('message-user') ? 'You' : 'Augmentum';
        const text = _clean(el.dataset.rawContent || el.querySelector('.message-content')?.textContent || '', 220);
        return text ? _item(role, text, { kind: 'message' }) : null;
      })
      .filter(Boolean);
    const voiceItems = this.voiceMessages.slice(-5).map((msg) => (
      _item(msg.role === 'user' ? 'You' : 'Augmentum', msg.text, { kind: 'voice' })
    ));
    const items = domMessages.length ? domMessages : voiceItems;
    const activeTitle = _queryText('#session-list .session-item.active .session-title, .session-item.active .session-title');
    this._set('chat', {
      title: 'Conversation',
      status: activeTitle || 'Live call',
      summary: items.length ? items[items.length - 1].detail : 'No visible transcript yet. Voice turns will appear here as they happen.',
      metrics: items.length ? [`${items.length} recent turns`] : [],
      items,
    });
  }

  _refreshAnalytical() {
    const phases = _visibleText('.reasoning-phase, .reasoning-step, .flow-step, .reasoning-summary', 5)
      .map((text, index) => _item(`Step ${index + 1}`, text, { kind: 'analysis' }));
    const lastAssistant = Array.from(document.querySelectorAll('#chat-messages .message-assistant'))
      .slice(-1)[0];
    const latest = _clean(lastAssistant?.dataset.rawContent || lastAssistant?.textContent || '', 220);
    this._set('analytical', {
      title: 'Analyze',
      status: phases.length ? 'Reasoning visible' : 'Ready',
      summary: latest || 'Ask a research or reasoning question; sources and reasoning state will show here.',
      items: phases.length ? phases : (latest ? [_item('Latest answer', latest, { kind: 'answer' })] : []),
    });
  }

  _refreshAgentic() {
    const snap = this.agenticSnapshot || {};
    const title = _clean(snap.title || _queryText('#task-title-label') || 'Build task', 90);
    const status = _clean(snap.status || document.getElementById('task-title-label')?.dataset?.status || '', 50);
    const progress = _num(snap.progress_pct ?? document.getElementById('task-progress-bar')?.style?.width?.replace('%', ''), 0);
    const steps = Array.isArray(snap.steps) ? snap.steps : [];
    const artifacts = Array.isArray(snap.artifacts) ? snap.artifacts : [];
    const domSteps = steps.length ? [] : Array.from(document.querySelectorAll('.pipeline-step')).slice(0, 5).map((row, index) => ({
      name: row.dataset.stepName || row.querySelector('.pipeline-step-label')?.textContent || `Step ${index + 1}`,
      status: row.dataset.stepStatus || row.querySelector('.pipeline-step-state')?.textContent || '',
    }));
    const items = (steps.length ? steps : domSteps).slice(0, 5).map((step, index) => (
      _item(step.name || `Step ${index + 1}`, step.state || step.status || '', { kind: 'step' })
    ));
    for (const artifact of artifacts.slice(0, 2)) {
      items.push(_item(artifact.title || 'Artifact', artifact.meta || artifact.download_url || '', { kind: 'artifact' }));
    }
    this._set('agentic', {
      title: 'Build',
      status: status || (items.length ? 'Active' : 'Ready'),
      summary: title,
      metrics: progress ? [`${Math.round(progress)}%`] : [],
      items,
    });
  }

  _refreshNarrative() {
    const snap = this.narrativeSnapshot || {};
    const scene = _clean(snap.scene || snap.sceneTitle || snap.title || _queryText('.narrative-session-title, .narrative-scene-title'), 100);
    const chars = Array.isArray(snap.characters) ? snap.characters : [];
    const items = chars.slice(0, 5).map((char) => (
      _item(char.name || char.id || 'Character', char.summary || char.status || char.role || '', { kind: 'character' })
    ));
    const domLines = _visibleText('.narrative-message, .narrative-beat, .scene-recap, .char-card', 4);
    for (const line of domLines) items.push(_item('Scene', line, { kind: 'scene' }));
    this._set('narrative', {
      title: 'Story',
      status: scene || 'Narrative',
      summary: _clean(snap.summary || scene || 'Current character and scene state will appear here.', 200),
      items,
    });
  }

  _refreshCoder() {
    const workspace = _queryText('#coder-workspace-select option:checked') || _queryText('#coder-files-title') || 'Workspace';
    const status = _queryText('#coder-status-text') || _queryText('#coder-status-detail') || '';
    const detail = _queryText('#coder-status-detail');
    const activeFile = _queryText('.coder-tab.active, #coder-editor-title, .coder-file-title');
    const rows = _visibleText('.coder-plan-step, .coder-review-item, .coder-approval-card, .coder-terminal-line, .coder-log-line', 5);
    const items = rows.map((row, index) => _item(index === 0 ? 'Current' : `Item ${index + 1}`, row, { kind: 'coder' }));
    if (activeFile && !items.some((i) => i.label === 'File')) items.unshift(_item('File', activeFile, { kind: 'file' }));
    this._set('coder', {
      title: 'Coder',
      status: status || 'Ready',
      summary: detail || workspace,
      items,
    });
  }

  _refreshBrowse() {
    const current = this.browseSnapshot || {};
    const title = _clean(current.title || _queryText('#browse-reader-title, .browse-reader-title, .article-title'), 90);
    const url = _clean(current.url || current.href || _queryText('#browse-url-input, #browse-search-input'), 120);
    const results = _visibleText('.browse-result-card, .search-result, .article-card, .browse-note-card', 5)
      .map((text, index) => _item(index === 0 ? 'Top result' : `Result ${index + 1}`, text, { kind: 'result' }));
    this._set('browse', {
      title: 'Browse',
      status: title || 'Ready',
      summary: title || url || 'Search results, open pages, and saved sources will appear here.',
      items: results,
    });
  }

  _refreshMediaFromState() {
    if (!this.mediaState) return;
    const title = _clean(this.mediaState.title || this.mediaState.name || this.mediaState.videoTitle || '', 90);
    if (!title) return;
    const detail = _clean(this.mediaState.channel || this.mediaState.artist || this.mediaState.status || '', 100);
    const progress = _num(this.mediaState.progress || this.mediaState.progressPct || 0, 0);
    this._set('media', {
      title: 'Media',
      status: 'Now playing',
      summary: title,
      metrics: progress ? [`${Math.round(progress <= 1 ? progress * 100 : progress)}%`] : [],
      items: [_item(title, detail, { kind: 'now-playing' })],
    });
  }

  async _refreshFiles() {
    const data = await _fetchJson('/api/files/search?limit=6&offset=0&sort=newest');
    const files = _firstArray(data, ['files', 'items', 'results']);
    this._set('files', {
      title: 'Files',
      status: `${files.length} recent`,
      summary: files.length ? 'Recent indexed files are ready to open, attach, compare, or summarize.' : 'No indexed files found yet.',
      items: files.map((f) => _item(f.name || f.title || f.filename, _fileDetail(f), { kind: _fileKind(f), id: f.id })),
    });
  }

  async _refreshNotes() {
    const data = await _fetchJson('/api/browse/notes');
    const notes = _firstArray(data, ['notes', 'items']).slice(0, MAX_ITEMS);
    this._set('notes', {
      title: 'Notes',
      status: `${notes.length} recent`,
      summary: notes.length ? 'Recent notes are available for dictation, clipping, and organization.' : 'No notes yet.',
      items: notes.map((n) => _item(n.title || 'Untitled note', [
        n.source_title || n.source_url || '',
        _dateish(n.updated_at || n.created_at),
      ].filter(Boolean).join(' | '), { kind: 'note', id: n.id })),
    });
  }

  async _refreshStudio() {
    const data = await _fetchJson('/api/artifacts');
    const artifacts = _firstArray(data, ['artifacts', 'items', 'results']).slice(0, MAX_ITEMS);
    this._set('studio', {
      title: 'Studio',
      status: `${artifacts.length} artifacts`,
      summary: artifacts.length ? 'Recent generated artifacts and media are ready to inspect or edit.' : 'No generated artifacts yet.',
      items: artifacts.map((a) => _item(
        a.display_name || a.filename || a.title || 'Artifact',
        _artifactDetail(a),
        { kind: a.format || a.metadata?.kind || 'artifact', id: a.id },
      )),
    });
  }

  async _refreshMedia() {
    const data = await _fetchJson('/api/files/search?limit=6&offset=0&sort=newest&media_status=in_progress');
    const media = _firstArray(data, ['files', 'items', 'results']);
    if (!media.length && this.mediaState) {
      this._refreshMediaFromState();
      return;
    }
    this._set('media', {
      title: 'Media',
      status: media.length ? 'Continue' : 'Ready',
      summary: media.length ? 'Continue watching, reading, or listening from your library.' : 'No in-progress media found.',
      items: media.map((m) => _item(m.name || m.title || 'Media item', _fileDetail(m), { kind: _fileKind(m), id: m.id })),
    });
  }

  async _refreshDevices() {
    const data = await _fetchJson('/api/devices');
    const devices = _firstArray(data, ['devices', 'items', 'results']);
    this._set('devices', {
      title: 'Devices',
      status: `${devices.length} device${devices.length === 1 ? '' : 's'}`,
      summary: devices.length ? 'Connected and discovered devices are available for cast or control.' : 'No connected devices discovered.',
      items: devices.map((d) => _item(
        d.name || d.label || d.id || d.host || 'Device',
        [d.type || d.driver || d.kind || '', d.status || d.state || '', d.host || d.address || ''].filter(Boolean).join(' | '),
        { kind: 'device', id: d.id },
      )),
    });
  }

  async _refreshGames() {
    const data = await _fetchJson('/api/files/search?limit=6&offset=0&sort=newest&kind=game');
    const games = _firstArray(data, ['files', 'items', 'results']);
    this._set('games', {
      title: 'Games',
      status: games.length ? 'Library' : 'Ready',
      summary: games.length ? 'Recent game entries are ready to launch or stream.' : 'No game entries found in the file library yet.',
      items: games.map((g) => _item(g.name || g.title || 'Game', _fileDetail(g), { kind: 'game', id: g.id })),
    });
  }
}

export function createXrSurfaceDataStore(options = {}) {
  return new XrSurfaceDataStore(options);
}
