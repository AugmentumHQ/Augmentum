/**
 * cast-app.js — TV cast surface for app + game artifacts.
 *
 * Reads ?id=<artifact_id> from the URL, fetches the full artifact,
 * routes to one of three render paths:
 *
 *   1. Embed-mode game (metadata.embed_src present)
 *      → set iframe.src directly. No bundle composition, no save bridge —
 *        embed-mode runs in its origin and writes to its own storage.
 *
 *   2. Local-bundle game (source_json.kind === "game_bundle")
 *      → composeBundle(entryHtml, files, entryPath, initialSave) into
 *        iframe.srcdoc. installSaveBridge pre-fetches + listens for
 *        storage-set / storage-remove / storage-clear postMessages and
 *        PUTs them back to /api/games/saves/{id}. Save state survives
 *        TV reload AND device handoff (Library play uses the same bridge).
 *
 *   3. App Builder application (source_json.type === "application")
 *      → assembleProject(files) into iframe.srcdoc. No save bridge (apps
 *        don't have the games saves backing store yet — adding it would
 *        match the games path 1:1 if that need surfaces).
 *
 * Anything else lands in the error overlay. The picker should have gated
 * non-cast-able artifacts before reaching this URL, so an error here is a
 * routing bug rather than a user mistake.
 *
 * Auth: this page boots inside the receiver's session (the stream-auth
 * redeem flow gave the TV browser a cookie). Same-origin fetches against
 * /api/artifacts/{id} and /api/games/saves/{id} flow through unchanged.
 */

import { composeBundle, installSaveBridge } from '../scripts/bundle-composer.js';
import { assembleProject } from '../scripts/assemble.js';

const params = new URLSearchParams(location.search);
const ARTIFACT_ID = (params.get('id') || '').trim();

const elFrame = document.querySelector('[data-cast-frame]');
const elBoot = document.querySelector('[data-cast-boot]');
const elTitle = document.querySelector('[data-cast-title]');
const elSub = document.querySelector('[data-cast-sub]');
const elError = document.querySelector('[data-cast-error]');
const elErrorTitle = document.querySelector('[data-cast-error-title]');
const elErrorDetail = document.querySelector('[data-cast-error-detail]');


function _hideBoot() {
  if (!elBoot) return;
  elBoot.setAttribute('data-hidden', '1');
  setTimeout(() => { elBoot.style.display = 'none'; }, 400);
}

function _showError(title, detail) {
  if (elBoot) elBoot.style.display = 'none';
  if (elError) {
    if (elErrorTitle) elErrorTitle.textContent = title || 'Can\'t cast this artifact';
    if (elErrorDetail) elErrorDetail.textContent = detail || '';
    elError.hidden = false;
  }
}

function _setTitle(name) {
  if (elTitle && name) elTitle.textContent = name;
  if (name) document.title = `Augmentum · ${name}`;
}


async function boot() {
  if (!ARTIFACT_ID) {
    _showError('Missing artifact id', 'Cast surface needs ?id=<artifact_id> in the URL.');
    return;
  }

  let artifact;
  try {
    const resp = await fetch(`/api/artifacts/${encodeURIComponent(ARTIFACT_ID)}`);
    if (!resp.ok) {
      _showError(
        resp.status === 404 ? 'Artifact not found' : 'Couldn\'t load artifact',
        resp.status === 403
          ? 'This cast session isn\'t authorised for that artifact.'
          : `HTTP ${resp.status} from /api/artifacts.`,
      );
      return;
    }
    artifact = await resp.json();
  } catch (err) {
    _showError('Network error', String(err?.message || err));
    return;
  }

  _setTitle(artifact?.display_name || artifact?.name || 'Casting');
  if (elSub) elSub.textContent = artifact?.metadata?.source || 'Casting from Augmentum';

  const meta = artifact?.metadata || {};
  let source = null;
  if (artifact?.source_json) {
    try {
      source = typeof artifact.source_json === 'string'
        ? JSON.parse(artifact.source_json)
        : artifact.source_json;
    } catch {
      source = null;
    }
  }

  // ── 1. Embed-mode game ──────────────────────────────────────────
  // Pin time resolved the framable URL into meta.embed_src; meta.embed_url
  // is the human source page (may carry frame-ancestors CSP). Prefer
  // embed_src; fall back to embed_url ONLY when nothing else fits — a CSP
  // failure beats a blank screen.
  if (meta.embed_src || meta.embed_url) {
    const url = meta.embed_src || meta.embed_url;
    elFrame.addEventListener('load', _hideBoot, { once: true });
    elFrame.src = url;
    return;
  }

  // Both remaining paths need a files manifest.
  const files = Array.isArray(source?.files) ? source.files : null;

  // ── 2. Local-bundle game ────────────────────────────────────────
  if (source?.kind === 'game_bundle' && files?.length) {
    const entryPath = source.entry || meta.bundle_entry || 'index.html';
    const entryFile = files.find(f => f.path === entryPath)
      || files.find(f => (f.path || '').toLowerCase().endsWith('.html'));
    if (!entryFile) {
      _showError('Game has no entry HTML', `Looked for ${entryPath} in the bundle.`);
      return;
    }
    let html = entryFile.content || '';
    if (entryFile.encoding === 'base64') {
      try { html = atob(html); } catch {
        _showError('Game entry can\'t decode', 'Base64 entry HTML failed to decode.');
        return;
      }
    }

    // Install the save bridge BEFORE setting srcdoc so the iframe's
    // storage-init postMessage lands on a live listener. Bridge state
    // also drives our initialSave pre-seed so a save written on the
    // Library Play surface is visible on the first read here.
    const bridge = installSaveBridge({ iframe: elFrame, artifactId: ARTIFACT_ID });
    let initialSave = {};
    try {
      initialSave = (await bridge.getInitialSave()) || {};
    } catch { /* empty save is fine — game starts fresh */ }

    const composed = composeBundle(html, files, entryFile.path, initialSave);
    elFrame.addEventListener('load', _hideBoot, { once: true });
    elFrame.srcdoc = composed;

    // Flush + uninstall when the receiver tab closes so the last second
    // of state isn't lost. The bridge's flush() is debounced internally;
    // the unload firing it once is enough.
    window.addEventListener('pagehide', () => bridge.uninstall(), { once: true });
    return;
  }

  // ── 3. App Builder application ──────────────────────────────────
  // App Builder shape uses role-tagged files (entry/style/script/module).
  // assembleProject is the same function the New Tab Play path uses, so
  // the TV gets identical output. No save bridge — apps don't write to
  // /api/games/saves and adding a parallel apps-saves backing is out of
  // scope here.
  if (source?.type === 'application' && files?.length) {
    const html = assembleProject(files);
    if (!html) {
      _showError('App has no entry file', 'The bundle is missing a role="entry" file.');
      return;
    }
    elFrame.addEventListener('load', _hideBoot, { once: true });
    elFrame.srcdoc = html;
    return;
  }

  // Nothing matched. Either the artifact isn't actually castable or its
  // source_json is malformed. The Library's cast picker should have gated
  // this, so an error here means a routing or pin-time bug — surface
  // enough detail that the user can report it.
  _showError(
    'This artifact can\'t be cast',
    `Type=${artifact?.format || '?'}, source_kind=${source?.kind || source?.type || 'unknown'}.`,
  );
}

boot();
