// ui/scripts/agent/launch-picker.js
//
// "Play solo or with a partner?" modal shown after a user clicks an
// emulator-ROM card. Two big buttons:
//
//   * Launch          → openEmulatorStage(artifact)
//   * Launch with Partner → openEmulatorStage(artifact, { startWithAgent: true, characterId })
//
// When the user picks Launch-with-Partner AND they have characters in
// their library, a second screen lets them pick which companion plays
// alongside. The choice is forwarded as ``character_id`` in the
// session POST, where the server loads the persona + voice from
// ``ui_characters`` and threads them into the slow-path prompt.
//
// For systems the agent doesn't yet support (no entry in
// agent-panel.js _SYSTEM_DEFAULTS) the picker bypasses itself and
// returns the solo shape immediately, so users on PSX / N64 / etc.
// don't see a meaningless choice.

import { isAgentSupported } from './agent-panel.js';


/**
 * Show the launch-mode chooser. Resolves to one of:
 *   { mode: 'solo' }
 *   { mode: 'partner', characterId: string | null }
 *   null   — user dismissed with Escape or backdrop click
 *
 * The "partner" return value always carries a ``characterId``:
 *   - non-empty string: a real character row id, the server will load
 *     name + persona + voice from ui_characters.
 *   - null: no character chosen (or library was empty / fetch failed).
 *     The session runs in anonymous companion mode — generic addendum,
 *     default voice.
 *
 * @param {object} opts
 *   @param {object} opts.artifact - title artifact (used for label + system lookup)
 *   @param {string} [opts.system] - libretro system id, if known. If
 *       omitted, the picker tries to read ``artifact.metadata.system``.
 *   @param {function} [opts.fetchImpl=fetch]
 * @returns {Promise<{mode:'solo'} | {mode:'partner', characterId: string|null} | null>}
 */
export function chooseLaunchMode({ artifact, system, fetchImpl }) {
  const fetchFn = fetchImpl || fetch;
  const sys = system || (artifact && artifact.metadata && artifact.metadata.system) || '';
  // Fast-path: nothing to choose if the system has no agent default.
  if (!isAgentSupported(sys)) {
    return Promise.resolve({ mode: 'solo' });
  }
  return new Promise((resolve) => {
    const backdrop = document.createElement('div');
    backdrop.className = 'launch-picker-backdrop';

    const card = document.createElement('div');
    card.className = 'launch-picker-card';
    card.setAttribute('role', 'dialog');
    card.setAttribute('aria-modal', 'true');
    card.setAttribute('aria-label', 'Choose how to launch');

    backdrop.appendChild(card);
    document.body.appendChild(backdrop);

    let resolved = false;
    const finish = (val) => {
      if (resolved) return;
      resolved = true;
      window.removeEventListener('keydown', onKey, true);
      backdrop.removeEventListener('click', onBackdropClick);
      if (backdrop.parentNode) backdrop.parentNode.removeChild(backdrop);
      resolve(val);
    };

    function onKey(e) {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        finish(null);
      }
    }
    function onBackdropClick(e) {
      if (e.target === backdrop) finish(null);
    }

    window.addEventListener('keydown', onKey, true);
    backdrop.addEventListener('click', onBackdropClick);

    // ── First screen: solo vs partner ─────────────────────────────
    _renderModeStep({
      card,
      artifact,
      onSolo: () => finish({ mode: 'solo' }),
      onPartner: () => {
        // Second screen: pick a companion. Falls through with
        // characterId=null when there are no characters available.
        _renderCompanionStep({
          card,
          fetchFn,
          onPick: (characterId) => finish({ mode: 'partner', characterId }),
          onBack: () => {
            _renderModeStep({
              card,
              artifact,
              onSolo: () => finish({ mode: 'solo' }),
              onPartner: () => finish({ mode: 'partner', characterId: null }),
            });
          },
        });
      },
    });
  });
}


// ── Step 1: solo vs partner ──────────────────────────────────────────

function _renderModeStep({ card, artifact, onSolo, onPartner }) {
  card.innerHTML = '';

  const titleEl = document.createElement('h2');
  titleEl.className = 'launch-picker-title';
  titleEl.textContent = (artifact && (artifact.title || artifact.name)) || 'Launch';

  const subtitle = document.createElement('p');
  subtitle.className = 'launch-picker-subtitle';
  subtitle.textContent = 'Play solo, or bring an AI partner along.';

  const choices = document.createElement('div');
  choices.className = 'launch-picker-choices';

  const soloBtn = _buildChoice({
    icon: '▶',
    title: 'Launch',
    blurb: 'Play by yourself. Standard emulator experience.',
    className: 'launch-picker-choice solo',
  });
  const partnerBtn = _buildChoice({
    icon: '✦',
    title: 'Launch with Partner',
    blurb: 'Bring a companion along. They watch, react, and (in co-pilot mode) play with you.',
    className: 'launch-picker-choice partner',
  });

  choices.appendChild(soloBtn);
  choices.appendChild(partnerBtn);

  const hint = document.createElement('p');
  hint.className = 'launch-picker-hint';
  hint.textContent = 'You can flip Off / Watch / Co-pilot anytime from the side panel.';

  card.appendChild(titleEl);
  card.appendChild(subtitle);
  card.appendChild(choices);
  card.appendChild(hint);

  soloBtn.addEventListener('click', onSolo);
  partnerBtn.addEventListener('click', onPartner);

  // Focus the Partner button by default so Enter = partner.
  requestAnimationFrame(() => partnerBtn.focus());
}


// ── Step 2: which companion ─────────────────────────────────────────

function _renderCompanionStep({ card, fetchFn, onPick, onBack }) {
  card.innerHTML = '';

  const back = document.createElement('button');
  back.type = 'button';
  back.className = 'launch-picker-back';
  back.textContent = '← Back';
  back.addEventListener('click', onBack);

  const titleEl = document.createElement('h2');
  titleEl.className = 'launch-picker-title';
  titleEl.textContent = 'Pick a Companion';

  const subtitle = document.createElement('p');
  subtitle.className = 'launch-picker-subtitle';
  subtitle.textContent = 'Choose who plays alongside you. Their persona drives the agent’s tone and voice.';

  const list = document.createElement('div');
  list.className = 'launch-picker-companions';

  const loading = document.createElement('p');
  loading.className = 'launch-picker-hint';
  loading.textContent = 'Loading characters…';
  list.appendChild(loading);

  card.appendChild(back);
  card.appendChild(titleEl);
  card.appendChild(subtitle);
  card.appendChild(list);

  // Default "anonymous helper" option is always available regardless
  // of whether the user has any characters. Built once and re-used.
  const defaultRow = _buildCompanionRow({
    name: 'Default Companion',
    blurb: 'Anonymous helper, default voice. No persona.',
    avatar: '',
    initials: '✦',
  });
  defaultRow.addEventListener('click', () => onPick(null));

  Promise.resolve(fetchFn('/api/characters/', { credentials: 'include' }))
    .then(r => r.ok ? r.json() : { characters: [] })
    .catch(() => ({ characters: [] }))
    .then((data) => {
      list.innerHTML = '';
      list.appendChild(defaultRow);

      const characters = Array.isArray(data) ? data : (data.characters || []);
      if (!characters.length) {
        const note = document.createElement('p');
        note.className = 'launch-picker-hint';
        note.style.marginTop = '12px';
        note.textContent = 'No characters in your library yet — create one from the Characters page to add a custom persona.';
        list.appendChild(note);
      } else {
        for (const ch of characters.slice(0, 30)) {
          const row = _buildCompanionRow({
            name: ch.name || 'Unnamed',
            blurb: _shortPersonaBlurb(ch),
            avatar: ch.avatar || '',
            initials: (ch.name || '?').slice(0, 1).toUpperCase(),
          });
          row.addEventListener('click', () => onPick(ch.id));
          list.appendChild(row);
        }
      }
      requestAnimationFrame(() => defaultRow.focus());
    });
}


function _shortPersonaBlurb(ch) {
  // Characters can carry a personality field at the top level (Augmentum's
  // ui_characters JSON normalizes to that). Fall back to description.
  const raw = (ch.personality || ch.description || '').trim();
  if (!raw) return 'Custom companion.';
  return raw.length > 120 ? raw.slice(0, 117).trimEnd() + '…' : raw;
}


function _buildCompanionRow({ name, blurb, avatar, initials }) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'launch-picker-companion';

  const avatarEl = document.createElement('span');
  avatarEl.className = 'launch-picker-companion-avatar';
  if (avatar && /^(https?:|data:image\/)/i.test(avatar)) {
    const img = document.createElement('img');
    img.src = avatar;
    img.alt = '';
    avatarEl.appendChild(img);
  } else {
    avatarEl.textContent = initials || '?';
  }

  const textWrap = document.createElement('span');
  textWrap.className = 'launch-picker-companion-text';
  const nameEl = document.createElement('span');
  nameEl.className = 'launch-picker-companion-name';
  nameEl.textContent = name;
  const blurbEl = document.createElement('span');
  blurbEl.className = 'launch-picker-companion-blurb';
  blurbEl.textContent = blurb;
  textWrap.appendChild(nameEl);
  textWrap.appendChild(blurbEl);

  btn.appendChild(avatarEl);
  btn.appendChild(textWrap);
  return btn;
}


function _buildChoice({ icon, title, blurb, className }) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = className;
  const iconEl = document.createElement('span');
  iconEl.className = 'launch-picker-choice-icon';
  iconEl.setAttribute('aria-hidden', 'true');
  iconEl.textContent = icon;
  const titleEl = document.createElement('span');
  titleEl.className = 'launch-picker-choice-title';
  titleEl.textContent = title;
  const blurbEl = document.createElement('span');
  blurbEl.className = 'launch-picker-choice-blurb';
  blurbEl.textContent = blurb;
  btn.appendChild(iconEl);
  btn.appendChild(titleEl);
  btn.appendChild(blurbEl);
  return btn;
}
