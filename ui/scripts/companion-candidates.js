/**
 * companion-candidates.js — clickable disambiguation cards for the
 * companion's media.play "offer" decision.
 *
 * When the resolver finds 2-4 plausible matches with no dominant one,
 * the server emits a ``companion.candidates`` surface event and Becca
 * speaks the options. This module renders them as cards docked above
 * the companion widget corner: cover-less but kind-badged, title +
 * subtitle, with an in-progress chip when a copy is partway through.
 *
 * Click → starts background playback through the same media.resume
 * routing the auto-play path uses (mini-player surfaces, no panel
 * yank), then the dock dismisses itself.
 *
 * Type-override is first-class: the dock NEVER traps focus or blocks
 * the composer. The server parks the same candidates in the
 * ReferentCache, so "the second one" typed or spoken resolves
 * server-side; when the resulting play dispatch fires, the
 * becca:verb-fired event auto-dismisses the dock.
 *
 * Companion Direct Action spec — docs/superpowers/specs/
 * 2026-06-10-companion-direct-action-design.md (L4 "offer").
 */

import { escapeHtml } from './app.js';
import { mediaCoverUrl, mediaBackdropUrl } from './files/api.js';
import { openCompanionDetail, closeCompanionDetail } from './companion-detail.js';

const HOST_ID = 'companion-candidates-dock';
const AUTO_DISMISS_MS = 90_000;   // stale options quietly leave

let _dismissTimer = null;

const _KIND_LABEL = {
  audiobook: 'Audiobook',
  podcast: 'Podcast',
  music: 'Music',
  video: 'Video',
  comic: 'Comic',
  book: 'Book',
  // Library-item candidates (game.play / game.recommend) — identified
  // by artifact_id instead of file_id; Play routes through
  // library/open-item.js rather than the media spine.
  game: 'Game',
  emulator_rom: 'Game',
  app: 'App',
  // Live TV channel candidates (livetv.play / livetv.browse) —
  // identified by channel_id + server_id; Watch routes through
  // /api/livetv/play. No file_index row; no artifact.
  live_tv: 'TV',
  // Coder workspace candidates (coder.delegate) — identified by
  // workspace_id (or is_new for the "New workspace" tile). Tapping enqueues
  // a background coder run rather than playing anything.
  coder_workspace: 'Workspace',
};

/**
 * Handle a coder-workspace pick: enqueue a background coder run in the chosen
 * workspace, or route "New workspace" into the Coder create UI (where the user
 * picks the template — never auto-select). ``delegation`` carries the build
 * prompt + resolved model strings; ``tierState`` carries the current
 * primary/heavyweight toggle.
 */
function _handleCoderPick(c, delegation, tierState) {
  const d = delegation || {};
  if (c.is_new || c.workspace_id === '__new__') {
    if (typeof window.openCoderNewWorkspace === 'function') {
      window.openCoderNewWorkspace({ prompt: d.prompt || '' });
    } else {
      window.__augmentum?.showToast?.('Open Coder to create a workspace.', 'info', 3000);
    }
    return;
  }
  const useHeavy = tierState?.tier === 'heavyweight' && d.heavyweight_model;
  const model = useHeavy ? d.heavyweight_model : (d.model || '');
  const prompt = d.prompt || '';
  if (!c.workspace_id || !prompt || !model) {
    window.__augmentum?.showToast?.('Could not start that build.', 'error', 3000);
    return;
  }
  fetch(`/api/coder/workspaces/${encodeURIComponent(c.workspace_id)}/background-runs`, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, model, coder_strategy: '' }),
  })
    .then(r => (r.ok ? r.json() : Promise.reject(new Error('enqueue failed'))))
    .then(() => {
      const tierNote = useHeavy ? ' (heavyweight)' : '';
      window.__augmentum?.showToast?.(
        `Building in ${c.title}${tierNote} — I'll bring you the result.`,
        'success', 3500,
      );
    })
    .catch(err => {
      console.warn('[candidates] coder delegate failed', err);
      window.__augmentum?.showToast?.('Could not start that build.', 'error', 3000);
    });
}

function _ensureStyles() {
  if (document.getElementById('companion-candidates-style')) return;
  const style = document.createElement('style');
  style.id = 'companion-candidates-style';
  style.textContent = `
    #${HOST_ID} {
      position: fixed;
      right: 18px;
      bottom: 96px;
      z-index: 880;
      width: min(330px, calc(100vw - 36px));
      display: flex;
      flex-direction: column;
      gap: 8px;
      animation: cc-rise 240ms ease-out;
    }
    @keyframes cc-rise {
      from { opacity: 0; transform: translateY(10px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    #${HOST_ID}.cc-leaving {
      transition: opacity 200ms ease, transform 200ms ease;
      opacity: 0;
      transform: translateY(8px);
      pointer-events: none;
    }
    .cc-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 12px;
      color: var(--text-secondary, #9aa0a6);
      padding: 0 4px;
    }
    .cc-header .cc-hint { opacity: 0.85; }
    .cc-close {
      background: none;
      border: none;
      color: inherit;
      font-size: 14px;
      line-height: 1;
      cursor: pointer;
      padding: 4px 6px;
      border-radius: 6px;
    }
    .cc-close:hover { background: color-mix(in srgb, currentColor 12%, transparent); }
    .cc-card {
      display: flex;
      align-items: center;
      gap: 10px;
      text-align: left;
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--border-color, rgba(255,255,255,0.09));
      background: var(--bg-elevated, var(--bg-secondary, #1d1f24));
      color: var(--text-primary, #e8eaed);
      cursor: pointer;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.18);
      transition: transform 120ms ease, border-color 120ms ease;
    }
    .cc-card:hover, .cc-card:focus-visible {
      transform: translateY(-1px);
      border-color: var(--accent-color, #7aa2f7);
      outline: none;
    }
    .cc-cover {
      flex: none;
      width: 40px;
      height: 56px;
      border-radius: 6px;
      object-fit: cover;
      background: color-mix(in srgb, currentColor 8%, transparent);
    }
    .cc-badge {
      flex: none;
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.4px;
      text-transform: uppercase;
      padding: 4px 7px;
      border-radius: 7px;
      color: var(--accent-color, #7aa2f7);
      background: color-mix(in srgb, var(--accent-color, #7aa2f7) 14%, transparent);
    }
    .cc-meta { min-width: 0; flex: 1; }
    .cc-title {
      font-size: 13px;
      font-weight: 600;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .cc-subtitle {
      font-size: 11.5px;
      color: var(--text-secondary, #9aa0a6);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      margin-top: 2px;
    }
    .cc-progress {
      font-size: 10.5px;
      color: var(--accent-color, #7aa2f7);
      margin-top: 2px;
    }
    .cc-header-actions { display: flex; align-items: center; gap: 6px; }
    .cc-tier-chip {
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 999px;
      border: 1px solid var(--border-color, rgba(255,255,255,0.14));
      background: transparent;
      color: var(--text-secondary, #9aa0a6);
      cursor: pointer;
      white-space: nowrap;
    }
    .cc-tier-chip:hover { border-color: var(--accent-color, #7aa2f7); }
    .cc-tier-chip.cc-tier-on {
      color: var(--accent-color, #7aa2f7);
      border-color: var(--accent-color, #7aa2f7);
      background: color-mix(in srgb, var(--accent-color, #7aa2f7) 14%, transparent);
    }
  `;
  document.head.appendChild(style);
}

/** Render (or replace) the candidate dock. */
export function showCandidates(payload) {
  const candidates = Array.isArray(payload?.candidates)
    ? payload.candidates.filter(c =>
        c && c.title
        && (c.file_id || c.artifact_id || c.channel_id || c.workspace_id || c.is_new))
    : [];
  if (!candidates.length) return;

  // Coder-delegation offer: cards enqueue a background build instead of
  // playing media. ``delegation`` carries the prompt + model strings; the
  // tier toggle lets the user escalate to a pinned heavyweight model.
  const isCoder = payload?.intent === 'coder.delegate'
    || candidates.some(c => (c.content_kind || c.kind) === 'coder_workspace');
  const delegation = payload?.delegation || {};
  const tierState = { tier: delegation.tier === 'heavyweight' ? 'heavyweight' : 'primary' };

  _ensureStyles();
  dismissCandidates({ instant: true });

  const host = document.createElement('div');
  host.id = HOST_ID;
  host.setAttribute('role', 'dialog');
  host.setAttribute('aria-label', 'Which one did you mean?');

  const cards = candidates.slice(0, 4).map((c, i) => {
    const kindLabel = _KIND_LABEL[c.content_kind] || _KIND_LABEL[c.kind] || 'Media';
    const progress = c.in_progress
      ? '<div class="cc-progress">In progress — resumes where you left off</div>'
      : '';
    const subtitle = c.subtitle
      ? `<div class="cc-subtitle">${escapeHtml(c.subtitle)}</div>`
      : '';
    // Real cover: an explicit cover_url wins (game/library candidates
    // supply their own art); file-backed candidates use the same cover
    // endpoint the Files/Media panels use. Broken/absent covers are
    // removed post-render (see below); the kind badge remains as the
    // fallback identity. Candidates with neither just show the badge.
    const coverSrc = c.cover_url
      || (c.file_id ? mediaCoverUrl(c.file_id, { size: 96 }) : '');
    const cover = coverSrc
      ? `<img class="cc-cover" src="${escapeHtml(coverSrc)}" alt="" loading="lazy">`
      : '';
    return `
      <button class="cc-card" data-idx="${i}">
        ${cover}
        <span class="cc-badge">${escapeHtml(kindLabel)}</span>
        <span class="cc-meta">
          <div class="cc-title">${escapeHtml(c.title)}</div>
          ${subtitle}
          ${progress}
        </span>
      </button>`;
  }).join('');

  const hint = isCoder ? 'Which workspace? Tap, or just say.' : 'Which one? Tap, or just say.';
  // Optional heavyweight escalation — only when the user has pinned a
  // heavyweight model in the model manager. Default is the primary chat model.
  const tierChip = (isCoder && delegation.heavyweight_available)
    ? `<button class="cc-tier-chip${tierState.tier === 'heavyweight' ? ' cc-tier-on' : ''}"
         aria-pressed="${tierState.tier === 'heavyweight'}"
         title="Use your pinned heavyweight model for this build">⚡ Heavyweight</button>`
    : '';
  host.innerHTML = `
    <div class="cc-header">
      <span class="cc-hint">${hint}</span>
      <span class="cc-header-actions">
        ${tierChip}
        <button class="cc-close" aria-label="Dismiss">✕</button>
      </span>
    </div>
    ${cards}
  `;

  const tierBtn = host.querySelector('.cc-tier-chip');
  if (tierBtn) {
    tierBtn.addEventListener('click', () => {
      tierState.tier = tierState.tier === 'heavyweight' ? 'primary' : 'heavyweight';
      tierBtn.classList.toggle('cc-tier-on', tierState.tier === 'heavyweight');
      tierBtn.setAttribute('aria-pressed', String(tierState.tier === 'heavyweight'));
    });
  }

  // Drop covers that 404 / aren't generated yet so the card falls back to
  // the badge cleanly instead of showing a broken-image glyph.
  host.querySelectorAll('.cc-cover').forEach(img => {
    img.addEventListener('error', () => img.remove(), { once: true });
  });
  host.querySelector('.cc-close')?.addEventListener('click', () => dismissCandidates());
  host.querySelectorAll('.cc-card').forEach(btn => {
    btn.addEventListener('click', () => {
      const c = candidates[Number(btn.dataset.idx)];
      if (!c) { dismissCandidates({ instant: true }); return; }
      if (isCoder) {
        // Coder pick → enqueue a background build (or route "new" to create).
        _handleCoderPick(c, delegation, tierState);
      } else {
        // Media pick → open the detail panel so the user can see what it is,
        // play it, or go BACK to the other picks (we stash the list for Back).
        _openCandidateDetail(c, candidates);
      }
      dismissCandidates({ instant: true });
    });
  });

  document.body.appendChild(host);

  clearTimeout(_dismissTimer);
  _dismissTimer = setTimeout(() => dismissCandidates(), AUTO_DISMISS_MS);
}

/** Remove the dock (with a soft exit unless instant). */
export function dismissCandidates({ instant = false } = {}) {
  clearTimeout(_dismissTimer);
  _dismissTimer = null;
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  if (instant) {
    host.remove();
    return;
  }
  host.classList.add('cc-leaving');
  setTimeout(() => host.remove(), 220);
}

const _KIND_LABEL_DETAIL = _KIND_LABEL;

function _fmtDuration(sec) {
  const s = Number(sec) || 0;
  if (s <= 0) return '';
  const h = Math.floor(s / 3600);
  const m = Math.round((s % 3600) / 60);
  return h ? `${h}h ${m}m` : `${m}m`;
}

/**
 * Full metadata chips for a file_index entry — every kind, every field
 * the sync layer actually populates. The source_metadata schema is
 * UNIFORM across media sources (Emby/JF, ABS, Suwayomi, LibriVox all
 * normalize to the same keys at sync: author/narrator/genres/year/
 * duration_s/series_name/season·episode/progress_pct/chapters/
 * library_name/unplayed_count/overview), so one renderer keyed on
 * what's present covers movies, shows, audiobooks, podcasts, comics,
 * and books alike. Empty fields are simply omitted — no blank chips.
 */
function _metaFieldsFor(entry, c) {
  const sm = entry.source_metadata || {};
  const ek = String(sm.entity_kind || '').toLowerCase();
  const fields = [];
  const add = (label, value) => {
    const v = String(value ?? '').trim();
    if (v) fields.push({ label, value: v });
  };

  add('', _fmtDuration(sm.duration_s));
  if (sm.year) add('Year', sm.year);
  if (sm.author) add(entry.kind === 'audio' ? 'Author' : 'By', sm.author);
  if (sm.narrator) add('Narrator', sm.narrator);
  const genres = Array.isArray(sm.genres) ? sm.genres.filter(Boolean) : [];
  if (genres.length) add('Genres', genres.slice(0, 3).join(', '));

  // Series placement — shows and book/comic series share the shape.
  if (sm.season_number != null && sm.episode_number != null) {
    add('', `S${sm.season_number} · E${sm.episode_number}`);
  } else if (sm.series_name && sm.series_sequence) {
    add('Series', `${sm.series_name} #${sm.series_sequence}`);
  } else if (sm.series_name && ek !== 'series') {
    add('Series', sm.series_name);
  }
  if (ek === 'series' && sm.unplayed_count) {
    add('', `${sm.unplayed_count} unplayed`);
  }

  const chapters = Array.isArray(sm.chapters) ? sm.chapters.length : 0;
  if (chapters > 1) add('', `${chapters} chapters`);

  // Progress — one honest chip, never both.
  if (sm.is_finished) {
    add('', 'Finished');
  } else {
    const pct = Number(sm.progress_pct) || 0;
    if (c?.in_progress || (pct > 0 && pct < 1)) {
      add('', pct > 0
        ? `${Math.round(pct * 100)}% in — resumes where you left off`
        : 'Resumes where you left off');
    }
  }

  if (sm.language) add('Language', sm.language);
  if (sm.library_name) add('Library', sm.library_name);
  if (sm.provider || entry.source) add('Source', sm.provider || entry.source);
  return fields;
}

/**
 * Open the generic companion detail panel for a picked candidate: show its
 * info, offer Play, and a Back that restores the recommendation list. Opens
 * immediately with the candidate's basic info, then enriches from the file
 * entry once it lands. If the file_id doesn't resolve, the panel says so
 * honestly instead of the old silent no-op.
 */
function _openCandidateDetail(c, allCandidates) {
  const badge = _KIND_LABEL_DETAIL[c.content_kind] || _KIND_LABEL_DETAIL[c.kind] || 'Media';
  const onBack = () => showCandidates({ candidates: allCandidates });

  // Live TV candidates (channels) — no file_index row, no artifact;
  // the channel lives on the user's Emby/JF server. Play routes through
  // the same POST /api/livetv/play path as clicking a tile in Files.
  if (!c.file_id && !c.artifact_id && c.channel_id && c.server_id) {
    const launch = () => {
      import('./files/live-tv-rails.js')
        .then(m => m.playLiveTvChannel({
          serverId: c.server_id, channelId: c.channel_id, name: c.title || '',
        }))
        .catch(err => {
          console.warn('[candidates] live TV launch failed', err);
          window.__augmentum?.showToast?.('Tune failed.', 'error', 3000);
        });
      closeCompanionDetail();
    };
    const fields = [];
    if (c.channel_number) {
      fields.push({ label: '', value: `Channel ${c.channel_number}` });
    }
    if (c.current_program) {
      fields.push({ label: 'Now', value: c.current_program });
    }
    openCompanionDetail({
      title: c.title,
      subtitle: c.subtitle || '',
      badge: _KIND_LABEL_DETAIL.live_tv || 'Live TV',
      coverUrl: c.cover_url || '',
      description: c.description || '',
      fields,
      onBack,
      actions: [{ label: 'Watch', primary: true, onClick: launch }],
    });
    return;
  }

  // Library-item candidates (games / ROMs / apps) — no file_index row
  // to enrich from; the artifact is the source of truth and the launch
  // path is the shared library dispatcher, not the media spine.
  if (!c.file_id && c.artifact_id) {
    const launch = () => {
      import('./library/open-item.js')
        .then(m => m.openLibraryItemById(c.artifact_id, { label: c.title || '' }))
        .catch(err => {
          console.warn('[candidates] game launch failed', err);
          window.__augmentum?.showToast?.('Launch failed.', 'error', 3000);
        });
      closeCompanionDetail();
    };
    // Full game metadata: system / author / source from the artifact,
    // play history (times played, total playtime, last played) joined
    // in server-side by the games verbs.
    const fields = [];
    const gAdd = (label, value) => {
      const v = String(value ?? '').trim();
      if (v) fields.push({ label, value: v });
    };
    gAdd('', c.system ? String(c.system).toUpperCase() : '');
    gAdd('By', c.author);
    gAdd('Source', c.source);
    if (c.runs != null) {
      if (Number(c.runs) === 0) {
        gAdd('', 'Never played');
      } else {
        gAdd('', `Played ${c.runs}×${c.playtime_s ? ` · ${_fmtDuration(c.playtime_s)} total` : ''}`);
        if (c.last_played) gAdd('Last played', String(c.last_played).slice(0, 10));
      }
    }
    openCompanionDetail({
      title: c.title,
      subtitle: c.subtitle || '',
      badge,
      coverUrl: c.cover_url || '',
      description: c.description || '',
      fields,
      onBack,
      actions: [{ label: 'Play', primary: true, onClick: launch }],
    });
    return;
  }

  // Route through the canonical off-surface opener, NOT raw activateFile —
  // the raw path silently no-ops for media-server rows without stream keys
  // and for anything the Files-grid resolver can't see when Files is
  // closed (the 2026-07-18 "Play movie just goes away" class). openContent
  // guarantees playback, a viewer, or an honest Files landing + toast.
  const _play = (entry) => {
    if (!entry) return;
    import('./files/open-content.js')
      .then(m => m.openContent(entry, { label: entry.name || c.title || '' }))
      .catch(err => {
        console.warn('[candidates] play failed', err);
        window.__augmentum?.showToast?.('Playback failed to start.', 'error', 3000);
      });
    closeCompanionDetail();
  };

  // Base descriptor from what the candidate already carries.
  const base = {
    title: c.title,
    subtitle: c.subtitle || '',
    badge,
    coverUrl: mediaCoverUrl(c.file_id, { size: 300 }),
    fields: c.in_progress ? [{ label: '', value: 'In progress — resumes where you left off' }] : [],
    onBack,
    actions: [{ label: 'Play', primary: true, disabled: true, onClick: () => {} }],
  };
  openCompanionDetail(base);

  // Enrich from the real file entry — backdrop, description, runtime, and a
  // working Play. This also surfaces the resolution problem: if the entry
  // 404s, we keep the basics and tell the user rather than doing nothing.
  fetch(`/api/files/entry/${encodeURIComponent(c.file_id)}`, { credentials: 'same-origin' })
    .then(r => (r.ok ? r.json() : null))
    .then(entry => {
      if (!entry || !entry.id) {
        openCompanionDetail({
          ...base,
          note: "I couldn't pull this one up from your library just now — the reference didn't resolve.",
          actions: [{ label: 'Play unavailable', disabled: true, onClick: () => {} }],
        });
        return;
      }
      const sm = entry.source_metadata || {};
      openCompanionDetail({
        title: entry.name || c.title,
        subtitle: sm.series_name || sm.author || c.subtitle || '',
        badge,
        coverUrl: mediaCoverUrl(entry.id, { size: 300 }),
        backdropUrl: sm.has_cover ? mediaBackdropUrl(entry.id) : '',
        description: entry.description || sm.overview || sm.description || '',
        fields: _metaFieldsFor(entry, c),
        onBack,
        actions: [{ label: 'Play', primary: true, onClick: () => _play(entry) }],
      });
    })
    .catch(err => {
      console.warn('[candidates] detail enrich failed', err);
      openCompanionDetail({
        ...base,
        note: "Couldn't load the full details right now.",
      });
    });
}

// A play verb landing by ANY route (typed follow-up resolved server-
// side, voice pick, another card) means the question is answered.
document.addEventListener('becca:verb-fired', (ev) => {
  const channel = ev?.detail?.channel || '';
  if (channel === 'media.resume' || channel === 'grove.play'
      || channel === 'game.launch' || channel === 'livetv.tune'
      // Coder delegation resolved by voice/typed pick ("the second one") or a
      // "new workspace" route — the question is answered, drop the dock.
      || channel === 'coder.delegate' || channel === 'coder.new_workspace') {
    dismissCandidates();
  }
});
