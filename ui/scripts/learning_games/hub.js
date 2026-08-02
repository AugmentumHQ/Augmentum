/**
 * Games hub — launches the learning games.
 *
 * The hub is a single modal that lists every game as a card with its
 * tagline and emoji. Each card calls its game's launcher, which takes
 * full control of the screen via a fullscreen overlay. On close, the
 * hub re-renders so the user can pick another game without leaving.
 */

import { escapeHtml } from '../app.js';
import { fetchBestScores, fetchGameReadiness, resolveVoiceForLang } from './_common.js';
import {
  openLanguagePartner,
  isPartnerSupported,
  langLabel,
} from './partner_launch.js';
import { launchBubblePop } from './bubble_pop.js';
import { launchWordGarden } from './word_garden.js';
import { launchEchoChamber } from './echo_chamber.js';
import { launchWhisperRace } from './whisper_race.js';
import { launchStoryWeaver } from './story_weaver.js';
import { launchWordForge } from './word_forge.js';
import { launchConstellation } from './constellation.js';
import { launchMirror } from './mirror.js';
import { launchVocabQuest } from './vocab_quest.js';

const GAMES = [
  { id: 'bubble_pop',          name: 'Bubble Pop',        emoji: '🫧', tagline: 'Pop the bubble that matches the spoken word.',  palette: 'cyan',    launcher: launchBubblePop,        minPool: 4 },
  { id: 'word_garden',         name: 'Word Garden',       emoji: '🌱', tagline: 'Tend a garden where each plant is a word.',     palette: 'green',   launcher: launchWordGarden,       minPool: 1 },
  { id: 'echo_chamber',        name: 'Echo Chamber',      emoji: '🎧', tagline: 'Hear it. Decode it. Pick its meaning.',        palette: 'purple',  launcher: launchEchoChamber,      minPool: 4 },
  { id: 'whisper_race',        name: 'Whisper Race',      emoji: '🎙️', tagline: 'Speak the word — your voice is the timer.',     palette: 'ember',   launcher: launchWhisperRace,      minPool: 4 },
  { id: 'story_weaver',        name: 'Story Weaver',      emoji: '📜', tagline: 'A branching story written around your words.', palette: 'amber',   launcher: launchStoryWeaver,      minPool: 6 },
  { id: 'word_forge',          name: 'Word Forge',        emoji: '🔨', tagline: 'Hammer roots together. Discover compounds.',    palette: 'iron',    launcher: launchWordForge,        minPool: 8 },
  { id: 'constellation',       name: 'Constellation',     emoji: '✨', tagline: 'Draw a sentence by connecting star-words.',     palette: 'night',   launcher: launchConstellation,    minPool: 6 },
  { id: 'mirror',              name: 'Mirror',            emoji: '🪞', tagline: 'Drag tiles to translate. Both ways.',            palette: 'teal',    launcher: launchMirror,           minPool: 6 },
  { id: 'vocab_quest',         name: 'Vocab Quest',       emoji: '🗺️', tagline: 'Vocab unlocks doors in a tiny adventure.',     palette: 'forest',  launcher: launchVocabQuest,       minPool: 6 },
];

const GAME_META = {
  bubble_pop: { track: 'core', skill: 'Listen', tone: 'Sprint' },
  echo_chamber: { track: 'core', skill: 'Listen', tone: 'Decode' },
  whisper_race: { track: 'core', skill: 'Speak', tone: 'Recall' },
  mirror: { track: 'core', skill: 'Translate', tone: 'Build' },
  vocab_quest: { track: 'core', skill: 'Meaning', tone: 'Quest' },
  word_garden: { track: 'coach', skill: 'Review', tone: 'Collection' },
  story_weaver: { track: 'discovery', skill: 'Read', tone: 'Story', allowDiscovery: true },
  word_forge: { track: 'discovery', skill: 'Morphology', tone: 'Combine' },
  constellation: { track: 'discovery', skill: 'Syntax', tone: 'Sentence' },
};

const HUB_SECTIONS = [
  { id: 'core', title: 'Daily Practice' },
  { id: 'discovery', title: 'Discovery' },
  { id: 'coach', title: 'Coach' },
];


let _open = false;

export async function openGamesHub({ lang, voice }) {
  if (_open) return;
  _open = true;
  document.getElementById('lg-hub')?.remove();

  const overlay = document.createElement('div');
  overlay.id = 'lg-hub';
  overlay.className = 'lg-overlay lg-hub-overlay';
  const close = () => {
    _open = false;
    overlay.remove();
    document.removeEventListener('keydown', onKey);
  };
  function onKey(e) { if (e.key === 'Escape') close(); }
  document.addEventListener('keydown', onKey);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

  // Decorate cards with the user's best score per game + resolve a
  // language-matched TTS voice (auto-picks a Japanese voice when the
  // lang is `ja` etc., falling back to the user's current voice if no
  // match is installed). Partner availability is probed in parallel so
  // the chip can render with the right action without a second hop.
  const [bests, resolvedVoice, partnerAvail, readiness] = await Promise.all([
    fetchBestScores(lang).catch(() => ({})),
    resolveVoiceForLang(lang, voice).catch(() => voice),
    _probePartnerAvailable(lang),
    fetchGameReadiness(lang).catch(() => null),
  ]);

  const hubStats = _renderHubStats(readiness, lang);
  const primaryAction = _renderPrimaryAction(readiness);
  const cards = _renderGameSections(GAMES, bests, readiness);

  // "Talk with a partner" — a curated conversation partner per
  // language, backed by the narrative pipeline. Distinct from the drill
  // games: this is a *place* (open-ended, persistent) rather than a
  // round (timed, scored). Renders above the grid so it reads as the
  // primary surface, with the games as drills underneath.
  const partnerChip = partnerAvail
    ? `
    <button type="button" class="lg-hub-partner" data-lang="${escapeHtml(lang || '')}">
      <div class="lg-hub-partner-emoji">💬</div>
      <div class="lg-hub-partner-body">
        <div class="lg-hub-partner-name">Talk with a partner</div>
        <div class="lg-hub-partner-tag">A persistent ${escapeHtml(langLabel(lang))} conversation tutor — recasts your mistakes, remembers your sessions, suggests drills when you're stuck.</div>
      </div>
      <div class="lg-hub-partner-cta">Open →</div>
    </button>`
    : '';

  // Curriculum-path manifest probe — quietly checks whether a curated
  // path ships for this language. If so, we surface a "Path" chip that
  // opens the path viewer (units + grammar drills + aux content).
  const pathManifest = await _probePathManifest(lang);
  const pathChip = pathManifest
    ? `
    <button type="button" class="lg-hub-path" data-lang="${escapeHtml(lang || '')}">
      <div class="lg-hub-path-emoji">🧭</div>
      <div class="lg-hub-path-body">
        <div class="lg-hub-path-name">Curriculum path</div>
        <div class="lg-hub-path-tag">${escapeHtml(_pathTag(pathManifest))}</div>
      </div>
      <div class="lg-hub-path-cta">Open →</div>
    </button>`
    : '';

  overlay.innerHTML = `
    <div class="lg-hub">
      <div class="lg-hub-head">
        <div>
          <div class="lg-hub-title">Learning</div>
          <div class="lg-hub-sub">${escapeHtml(langLabel(lang))} practice</div>
        </div>
        <button type="button" class="lg-hub-close" aria-label="Close">×</button>
      </div>
      ${hubStats}
      ${primaryAction}
      ${partnerChip}
      ${pathChip}
      ${cards}
    </div>`;
  document.body.appendChild(overlay);

  overlay.querySelector('.lg-hub-close').addEventListener('click', close);

  // Partner chip → materialise the partner card, switch to narrative
  // mode, and open chat with that character. The hub closes itself so
  // the user lands in the chat surface, not on top of the modal.
  const partnerBtn = overlay.querySelector('.lg-hub-partner');
  if (partnerBtn) {
    partnerBtn.addEventListener('click', async () => {
      partnerBtn.classList.add('lg-hub-partner-launching');
      try {
        await openLanguagePartner(lang);
        close();
      } catch (err) {
        console.warn('[games] partner launch failed', err);
        partnerBtn.classList.remove('lg-hub-partner-launching');
      }
    });
  }

  // Curriculum-path chip → open the path viewer on top of the hub so
  // closing it returns the user to the games list. Distinct from the
  // partner chip (which closes the hub on launch) because the path
  // viewer is a peer-level reference, not a destination.
  const pathBtn = overlay.querySelector('.lg-hub-path');
  if (pathBtn) {
    pathBtn.addEventListener('click', () => _openPathViewer(lang));
  }

  overlay.querySelectorAll('[data-game]').forEach(btn => {
    btn.addEventListener('click', async () => {
      if (btn.disabled || btn.getAttribute('aria-disabled') === 'true') return;
      const g = GAMES.find(x => x.id === btn.dataset.game);
      if (!g) return;
      btn.classList.add('lg-hub-card-launching');
      try {
        await g.launcher({ lang, voice: resolvedVoice });
      } catch (err) {
        console.warn('[games] launcher failed', g.id, err);
      } finally {
        btn.classList.remove('lg-hub-card-launching');
      }
    });
  });
}

// Hub rendering helpers

function _readinessTotals(readiness) {
  const counts = readiness?.counts || {};
  return {
    total: Number(readiness?.total || 0),
    due: Number(readiness?.due || 0),
    weak: Number(readiness?.weak || 0),
    settled: Number(readiness?.settled || 0),
    learning: Number(counts.learning || 0),
    leech: Number(counts.leech || 0),
  };
}

function _statusForGame(g, readiness) {
  const meta = GAME_META[g.id] || {};
  const server = readiness?.games?.[g.id];
  if (server) {
    const rawProgress = Number(server.progress);
    return {
      ready: !!server.ready,
      progress: Number.isFinite(rawProgress) ? Math.max(0, Math.min(1, rawProgress)) : (server.ready ? 1 : 0),
      label: server.label || (server.ready ? 'Ready' : 'Locked'),
      min: Number(g.minPool || 1),
      meta,
      reason: server.reason || '',
      note: server.note || '',
      recommended: server.recommended !== false,
      requirements: server.requirements || {},
    };
  }

  const totals = _readinessTotals(readiness);
  const min = Number(g.minPool || 1);
  const discovery = meta.allowDiscovery || (meta.track === 'discovery' && g.id === 'story_weaver');
  const ready = discovery || totals.total >= min;
  const progress = min <= 0 ? 1 : Math.max(0, Math.min(1, totals.total / min));
  const label = ready
    ? (meta.allowDiscovery ? 'Explore' : (totals.due > 0 && meta.track === 'core' ? `${totals.due} due` : 'Ready'))
    : `${totals.total}/${min} words`;
  return { ready, progress, label, min, meta, reason: '', note: '', recommended: ready, requirements: {} };
}

function _renderHubStats(readiness, lang) {
  const totals = _readinessTotals(readiness);
  return `
    <section class="lg-hub-hero" aria-label="${escapeHtml(langLabel(lang))} progress">
      <div class="lg-hub-hero-main">
        <div class="lg-hub-hero-kicker">Today</div>
        <div class="lg-hub-hero-title">${totals.due > 0 ? `${totals.due} reviews waiting` : 'Practice is clear'}</div>
      </div>
      <div class="lg-hub-stats">
        <div class="lg-hub-stat"><b>${totals.total}</b><span>words</span></div>
        <div class="lg-hub-stat"><b>${totals.settled}</b><span>settled</span></div>
        <div class="lg-hub-stat"><b>${totals.weak}</b><span>weak</span></div>
      </div>
    </section>`;
}

function _pickPrimary(readiness) {
  const totals = _readinessTotals(readiness);
  const preferred = totals.due > 0
    ? ['bubble_pop', 'echo_chamber', 'whisper_race']
    : ['mirror', 'vocab_quest', 'word_garden', 'story_weaver'];
  return preferred.map(id => GAMES.find(g => g.id === id)).find(g => g && _statusForGame(g, readiness).ready)
    || GAMES.find(g => g.id === 'story_weaver');
}

function _renderPrimaryAction(readiness) {
  const g = _pickPrimary(readiness);
  if (!g) return '';
  const status = _statusForGame(g, readiness);
  const label = status.meta.allowDiscovery ? 'Start discovery' : 'Start practice';
  return `
    <button type="button" class="lg-hub-primary" data-game="${escapeHtml(g.id)}" data-palette="${escapeHtml(g.palette)}" ${status.ready ? '' : 'disabled aria-disabled="true"'}>
      <span class="lg-hub-primary-icon">${g.emoji}</span>
      <span class="lg-hub-primary-copy">
        <span class="lg-hub-primary-label">${escapeHtml(label)}</span>
        <strong>${escapeHtml(g.name)}</strong>
      </span>
      <span class="lg-hub-primary-cta">Go</span>
    </button>`;
}

function _renderGameSections(games, bests, readiness) {
  let i = 0;
  return `<div class="lg-hub-sections">${HUB_SECTIONS.map((section) => {
    const group = games.filter(g => (GAME_META[g.id]?.track || 'core') === section.id);
    if (!group.length) return '';
    const cards = group.map((g) => _renderGameCard(g, i++, bests[g.id] || {}, readiness)).join('');
    return `
      <section class="lg-hub-section" data-section="${escapeHtml(section.id)}">
        <div class="lg-hub-section-head">
          <h3>${escapeHtml(section.title)}</h3>
        </div>
        <div class="lg-hub-grid">${cards}</div>
      </section>`;
  }).join('')}</div>`;
}

function _renderGameCard(g, i, stats, readiness) {
  const status = _statusForGame(g, readiness);
  const meta = status.meta;
  const badge = stats.best
    ? `<div class="lg-hub-card-best">best <b>${escapeHtml(String(stats.best))}</b> &middot; ${escapeHtml(String(stats.plays || 0))} ${(stats.plays || 0) === 1 ? 'play' : 'plays'}</div>`
    : '';
  const disabled = status.ready ? '' : 'disabled aria-disabled="true"';
  const title = status.reason || status.note || g.tagline;
  return `
    <button type="button" class="lg-hub-card${status.ready ? '' : ' lg-hub-card-locked'}" data-game="${escapeHtml(g.id)}" data-palette="${escapeHtml(g.palette)}" data-track="${escapeHtml(meta.track || 'core')}" style="--lg-i: ${i}" title="${escapeHtml(title)}" ${disabled}>
      <div class="lg-hub-card-top">
        <div class="lg-hub-card-emoji">${g.emoji}</div>
        <div class="lg-hub-card-pill">${escapeHtml(status.label)}</div>
      </div>
      <div class="lg-hub-card-name">${escapeHtml(g.name)}</div>
      <div class="lg-hub-card-skill">${escapeHtml(meta.skill || 'Practice')} &middot; ${escapeHtml(meta.tone || 'Round')}</div>
      <div class="lg-hub-card-tag">${escapeHtml(g.tagline)}</div>
      <div class="lg-hub-card-progress" aria-hidden="true"><span style="width:${Math.round(status.progress * 100)}%"></span></div>
      ${badge}
    </button>`;
}

// ── Partner support ─────────────────────────────────────────────────
//
// Helper canon lives in ./partner_launch.js (shared with the homepage
// re-entry card). _probePartnerAvailable here is a thin wrapper that
// keeps the hub's existing call signature stable while delegating the
// supported-langs check to the shared module.


async function _probePartnerAvailable(lang) {
  return isPartnerSupported(lang);
}

// ── Curriculum-path probe + viewer ─────────────────────────────────
// `/api/learning/paths/{lang}/manifest` returns a path summary + the
// list of aux kinds (grammar, kanji, tones, characters, assessment)
// that ship for this language. If the language has no curated path,
// the route 404s and the chip stays hidden. The viewer is a separate
// modal so it doesn't compete with the games hub for attention.
async function _probePathManifest(lang) {
  if (!lang) return null;
  try {
    const r = await fetch(`/api/learning/paths/${encodeURIComponent(lang)}/manifest`, {
      credentials: 'same-origin',
    });
    if (!r.ok) return null;
    return await r.json();
  } catch (_) {
    return null;
  }
}

function _pathTag(manifest) {
  const summary = manifest?.path || manifest?.summary || {};
  const aux = Array.isArray(manifest?.aux_available)
    ? manifest.aux_available
    : (Array.isArray(manifest?.aux) ? manifest.aux : []);
  const levels = (summary.levels || []).length;
  const auxLabel = aux.length ? ` / ${aux.join(' / ')}` : '';
  return `${levels} level${levels === 1 ? '' : 's'}${auxLabel}`;
}

async function _openPathViewer(lang) {
  document.getElementById('lg-path-viewer')?.remove();

  const overlay = document.createElement('div');
  overlay.id = 'lg-path-viewer';
  overlay.className = 'lg-overlay lg-hub-overlay';
  overlay.innerHTML = `
    <div class="lg-hub" style="max-width:880px">
      <div class="lg-hub-head">
        <div>
          <div class="lg-hub-title">Curriculum path</div>
          <div class="lg-hub-sub">${escapeHtml(langLabel(lang))}</div>
        </div>
        <button type="button" class="lg-hub-close" aria-label="Close">×</button>
      </div>
      <div id="lg-path-body" style="overflow-y:auto;max-height:65vh">
        <div style="padding:var(--space-md);color:var(--text-muted)">Loading…</div>
      </div>
    </div>`;
  document.body.appendChild(overlay);

  const close = () => overlay.remove();
  overlay.querySelector('.lg-hub-close').addEventListener('click', close);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

  const body = overlay.querySelector('#lg-path-body');
  try {
    const r = await fetch(`/api/learning/paths/${encodeURIComponent(lang)}/manifest`, {
      credentials: 'same-origin',
    });
    if (!r.ok) {
      body.innerHTML = `<div style="padding:var(--space-md)">No curriculum for ${escapeHtml(langLabel(lang))} yet.</div>`;
      return;
    }
    const manifest = await r.json();
    const summary = manifest.path || manifest.summary || {};
    const aux = Array.isArray(manifest.aux_available)
      ? manifest.aux_available
      : (Array.isArray(manifest.aux) ? manifest.aux : []);
    const levels = summary.levels || [];

    const auxButtons = aux.map(kind => `
      <button class="btn btn-sm" data-aux="${escapeHtml(kind)}">${escapeHtml(kind)}</button>
    `).join('');

    const levelBlocks = levels.map(lvl => `
      <div style="margin-top:var(--space-md)">
        <div style="font-weight:600">${escapeHtml(lvl.code || '?')} - ${escapeHtml(lvl.name || lvl.label || '')}</div>
        <div class="lg-path-units" style="display:flex;flex-wrap:wrap;gap:var(--space-xs);margin-top:var(--space-xs)">
          ${(lvl.units || []).map(u => `
            <button class="btn btn-sm" data-unit-id="${escapeHtml(u.id)}" title="${escapeHtml(u.title || '')}">
              ${escapeHtml(u.title || u.id)}
            </button>
          `).join('')}
        </div>
      </div>
    `).join('');

    body.innerHTML = `
      <div style="padding:var(--space-md)">
        ${aux.length ? `<div style="margin-bottom:var(--space-sm)"><strong>Aux content:</strong> ${auxButtons}</div>` : ''}
        ${levelBlocks || '<div style="color:var(--text-muted)">No levels in this path yet.</div>'}
        <pre id="lg-path-detail" style="margin-top:var(--space-md);padding:var(--space-sm);background:var(--bg-elevated);border-radius:var(--radius-sm);font-size:12px;display:none;max-height:300px;overflow-y:auto"></pre>
      </div>`;

    const detail = body.querySelector('#lg-path-detail');
    body.querySelectorAll('[data-unit-id]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const unitId = btn.dataset.unitId;
        if (!unitId) return;
        detail.style.display = 'block';
        detail.textContent = 'Loading…';
        const r2 = await fetch(
          `/api/learning/paths/${encodeURIComponent(lang)}/unit/${encodeURIComponent(unitId)}`,
          { credentials: 'same-origin' },
        );
        detail.textContent = r2.ok
          ? JSON.stringify(await r2.json(), null, 2)
          : `Failed (status ${r2.status})`;
      });
    });
    body.querySelectorAll('[data-aux]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const kind = btn.dataset.aux;
        if (!kind) return;
        detail.style.display = 'block';
        detail.textContent = 'Loading…';
        const r2 = await fetch(
          `/api/learning/paths/${encodeURIComponent(lang)}/aux/${encodeURIComponent(kind)}`,
          { credentials: 'same-origin' },
        );
        detail.textContent = r2.ok
          ? JSON.stringify(await r2.json(), null, 2)
          : `Failed (status ${r2.status})`;
      });
    });
  } catch (err) {
    body.innerHTML = `<div style="padding:var(--space-md)">Failed: ${escapeHtml(String(err.message || err))}</div>`;
  }
}
