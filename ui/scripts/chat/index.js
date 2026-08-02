/* ==========================================================================
   Chat Module — Public API
   Orchestrates the chat system. Provides backward-compatible API surface
   so existing code that imports from chat.js continues to work unchanged.
   ========================================================================== */

import { app, escapeHtml, showToast, refreshDocContextBar, refreshDocumentList } from '../app.js';
import { addToRecentModels, getSettings } from '../settings.js';
import { getModelsSync } from '../model-cache.js';
import { restoreReasoningFromStored, resetReasoningState } from '../analytical.js';
import { onExecutionStart, onPhaseUpdate, onExecutionComplete, resetFlowBar, getTuneOverrides, clearTuneOverrides, getCurrentFlow } from '../flow-bar.js';
import { showAgenticTaskStarting, showAgenticTaskView } from '../agentic.js';

import { icons, STORAGE_ACTIVE } from './constants.js';
import * as tree from './tree.js';
import { renderMarkdown, highlightCode } from './markdown.js';
import { sessionStore } from './sessions.js';
import { MessageRenderer } from './renderer.js';
import { ChatInput } from './input.js';
import { ChatStream } from './stream.js';
import * as tts from './tts.js';
import { extractMotionCue } from '../motion-cue.js';
import {
  initCodeBlockActions, closePreviewModal, restoreCodeVersions,
  toggleHtmlPreview, runPythonCode, runCodeBlock, toggleSvgPreview, toggleCodeEdit,
  downloadCodeBlock, showVersion, getVersionIdx, getBlock,
  updateBlock, hydrateCodeBlocks, regenerateMarkdown, getSessionNode,
  updateVersionIndicator, getBlocksForMessage, getOutputPanel,
  appendOutput, appendError, renderExecutionOutput, getPythonPreamble,
  codeMindValidate, getPendingBlockEdits, clearPendingBlockEdits,
  getFixRetryCount, setSessionAccessor,
} from './code-actions.js';
import {
  showAskAiPrompt, executeAiEdit, showQuickActionsMenu,
  autoFixCodeBlock, silentLint, applyDiffPatches,
  createStreamingDiff, parseMultiBlockResponse,
  computeLineDiff, renderDiffLines,
  fixHTML, fixCSS, fixPython, fixJSON,
} from './code-edit.js';
import { initIllustrateMoment, setIllustrateSessionGetter, setIllustrateModeGetter } from './illustrate.js';
import { initMicButton as initMicButtonModule, isPendingVoiceInput, clearPendingVoiceInput } from './stt.js';
import * as memoryGlow from './memory-glow.js';
import { handleProjectResult, renderProjectCard, setChatBridge } from './project.js';
import { handleBuildStarted, recoverBuildCards } from './build-card.js';
import { runImpersonate, fetchActivePersona } from './impersonate.js';
import { mountOfferFeed } from './offer-chip.js';
import * as multiModel from './multi-model.js';

// Re-export modules for external consumers
export { icons } from './constants.js';
export { sessionStore } from './sessions.js';
export { MessageRenderer } from './renderer.js';
export { ChatInput } from './input.js';
export { ChatStream } from './stream.js';
export * from './tree.js';
export { renderMarkdown, highlightCode } from './markdown.js';
export { TtsSentenceBuffer, TtsAudioPipeline, ttsProgressiveFeed, ttsProgressiveFinish,
         ttsProgressiveCancel, ttsStopCurrent, ttsChatWarmup, ttsCleanText, ttsSplitChunks,
         ttsFetchAudio, ttsPlayBlob, ttsPlayMessage, ttsQueueAutoRead, ttsProcessQueue,
         setActiveSessionGetter, setCharacterVoiceLookup, _installActivePipeline } from './tts.js';

// Re-export code-actions and code-edit for external consumers
export {
  initCodeBlockActions, closePreviewModal, restoreCodeVersions,
  toggleHtmlPreview, runPythonCode, runCodeBlock, toggleSvgPreview, toggleCodeEdit,
  downloadCodeBlock, showVersion, getVersionIdx, getBlock,
  updateBlock, hydrateCodeBlocks, regenerateMarkdown, getSessionNode,
  updateVersionIndicator, getBlocksForMessage, getOutputPanel,
  appendOutput, appendError, renderExecutionOutput, getPythonPreamble,
  codeMindValidate, getPendingBlockEdits, clearPendingBlockEdits,
  getFixRetryCount,
} from './code-actions.js';
export {
  showAskAiPrompt, executeAiEdit, showQuickActionsMenu,
  autoFixCodeBlock, silentLint, applyDiffPatches,
  createStreamingDiff, parseMultiBlockResponse,
  computeLineDiff, renderDiffLines,
  fixHTML, fixCSS, fixPython, fixJSON,
} from './code-edit.js';

// Re-export new modules
export { initIllustrateMoment } from './illustrate.js';
export { initMicButton } from './stt.js';
export { memGlowInit, memGlowRecalling, memGlowIdle, memGlowLearned,
         memStartPolling, memStopPolling, memMarkExtracting } from './memory-glow.js';
export { scanLorebook } from './tree.js';

function getPrimaryInputArea() {
  return document.getElementById('chat-input')?.closest('.input-area') || null;
}

function _parseArtifactSource(info) {
  const raw = info?.source_json ?? info?.source ?? null;
  if (!raw) return null;
  if (typeof raw === 'string') {
    try { return JSON.parse(raw); } catch { return null; }
  }
  return typeof raw === 'object' ? raw : null;
}

function _isApplicationArtifact(info) {
  const source = _parseArtifactSource(info);
  return !!(source && source.type === 'application');
}

async function _openArtifactWorkspace(artifactId, mode = 'play') {
  // Workspace is a top-level overlay since the library-shell refactor —
  // it reparents itself to <body> on first openWorkspace(). Library is
  // opened too so the Back button inside the workspace lands the user
  // on a real Library surface rather than the chat backdrop.
  const mod = await import('../workspace.js');
  await mod.openWorkspace({ id: artifactId }, mode);
  import('../library.js').then(lib => lib.openLibrary()).catch(() => {});
}

async function _openArtifactPreview(artifactId) {
  if (!artifactId) return;
  try {
    const resp = await fetch(`/api/artifacts/${encodeURIComponent(artifactId)}`);
    if (resp.ok) {
      const info = await resp.json();
      if (_isApplicationArtifact(info)) {
        await _openArtifactWorkspace(artifactId, 'play');
        return;
      }
    }
  } catch { /* fall through to Studio overview */ }
  const m = await import('../studio.js');
  if (m.openStudio) {
    await m.openStudio(artifactId, { mode: 'overview' });
    return;
  }
  window.open(`/api/artifacts/${encodeURIComponent(artifactId)}/preview`, '_blank', 'noopener');
}

async function _openArtifactEditor(artifactId) {
  if (!artifactId) return;
  try {
    const resp = await fetch(`/api/artifacts/${encodeURIComponent(artifactId)}`);
    if (resp.ok) {
      const info = await resp.json();
      if (_isApplicationArtifact(info)) {
        await _openArtifactWorkspace(artifactId, 'work');
        return;
      }
    }
  } catch { /* fall back to Studio */ }

  const m = await import('../studio.js');
  if (m.openStudio) m.openStudio(artifactId);
}

function _agenticArtifactPayloadToCard(payload) {
  if (!payload || typeof payload !== 'object') return null;
  const artifactId = String(payload.artifact_id || payload.id || '');
  const title = String(payload.title || payload.display_name || payload.name || 'Artifact');
  const rawCard = payload.card && typeof payload.card === 'object' ? payload.card : null;
  const format = String(payload.format || payload.kind || '').toLowerCase();
  const pageType = String(payload.page_type || '').replace(/_/g, ' ').trim();
  const subtitle = pageType || (format && format !== 'artifact' ? format.toUpperCase() : '');
  const sizeBytes = Number(payload.size_bytes) || 0;
  const downloadUrl = payload.download_url || (artifactId ? `/api/artifacts/${encodeURIComponent(artifactId)}/download` : '');

  const card = rawCard ? {
    ...rawCard,
    preview: rawCard.preview && typeof rawCard.preview === 'object' ? { ...rawCard.preview } : (rawCard.preview || {}),
    actions: Array.isArray(rawCard.actions)
      ? rawCard.actions.map((action) => (
        action && typeof action === 'object'
          ? {
            ...action,
            payload: action.payload && typeof action.payload === 'object'
              ? { ...action.payload }
              : action.payload,
          }
          : action
      ))
      : [],
  } : {
    kind: 'artifact',
    title,
    subtitle,
    summary: '',
    preview: {},
    actions: [],
  };

  const rawKind = rawCard && rawCard.kind ? String(rawCard.kind) : 'artifact';
  card.kind = ['artifact', 'image', 'search', 'article', 'code_exec', 'calc', 'data'].includes(rawKind)
    ? rawKind
    : 'artifact';
  card.artifact_id = String(card.artifact_id || artifactId);
  if (!card.title) card.title = title;
  if (!card.subtitle && subtitle) card.subtitle = subtitle;
  if (!card.preview || typeof card.preview !== 'object') card.preview = {};
  if (format && !card.preview.format) card.preview.format = format;
  if (sizeBytes && !card.preview.size_bytes) card.preview.size_bytes = sizeBytes;

  if (!Array.isArray(card.actions) || !card.actions.length) {
    card.actions = [
      {
        label: 'Preview',
        event: 'artifact:preview',
        payload: { artifact_id: card.artifact_id },
        icon: 'eye',
      },
      {
        label: 'Edit',
        event: 'artifact:edit',
        payload: { artifact_id: card.artifact_id },
        icon: 'edit',
      },
      ...(downloadUrl ? [{
        label: 'Download',
        href: downloadUrl,
        icon: 'download',
      }] : []),
    ];
  }

  return card;
}

function _pushPendingAgenticArtifactCard(card) {
  if (!card) return;
  const artifactId = String(card.artifact_id || '');
  if (artifactId) {
    const existing = _pendingAgenticArtifactCards.findIndex((entry) => String(entry?.artifact_id || '') === artifactId);
    if (existing >= 0) _pendingAgenticArtifactCards.splice(existing, 1);
  }
  _pendingAgenticArtifactCards.push(card);
}

// ---------------------------------------------------------------------------
// ChatModule — singleton orchestrator
// ---------------------------------------------------------------------------

/**
 * The primary surface's instances. When chat.js acts as a shim,
 * it delegates to these. In multi-surface mode, each ChatSurface
 * creates its own renderer/input/stream via the classes directly.
 */
let _primaryRenderer = null;
let _primaryInput = null;
let _activeStream = null;   // Currently active ChatStream for the primary surface
let _initialized = false;
// One-shot auto-resume on network drop. Reset at the start of each
// fresh (non-continue) turn so a single user message can use its one
// retry without piling up retries forever if the backend really is down.
let _autoResumeAttempted = false;

// Agentic step expand/collapse tracking
const agenticExpandedSteps = new Set();

/**
 * Initialize the chat system.
 * Called by app.js during startup. Sets up the session store,
 * creates the primary surface instances, and wires global event listeners.
 */
/**
 * Snap the primary chat scroll area to the bottom. Called by surfaces
 * when they activate so tabbing back to the primary chat / narrative
 * tab lands on the latest message instead of the top — the browser
 * discards scroll position when the surface-container flips between
 * opacity 0 and 1 (equivalent to display:none from the layout engine's
 * perspective). Two rAFs let browser layout settle before we snap.
 */
export function scrollPrimaryToBottom() {
  if (!_primaryRenderer) return;
  _primaryRenderer.scrollToBottom(false, true);
  requestAnimationFrame(() => {
    _primaryRenderer?.scrollToBottom(false, true);
    requestAnimationFrame(() => _primaryRenderer?.scrollToBottom(false, true));
  });
}

export async function initChat() {
  if (_initialized) return;
  _initialized = true;

  // Load sessions from server
  await sessionStore.load();

  // Restore active session for current mode
  const currentMode = app.state.mode || 'passthrough';
  const activeId = sessionStore.getActiveId();
  const activeSession = activeId ? sessionStore.get(activeId) : null;
  if (activeSession && (activeSession.mode || 'passthrough') === currentMode) {
    // Active session matches current mode — keep it
  } else {
    // No active session, or it's for a different mode — pick the most recent
    const modeSessions = sessionStore.forMode(currentMode);
    if (modeSessions.length > 0) {
      app.state.currentSessionId = modeSessions[0].id;
      sessionStore.setActiveId(modeSessions[0].id);  // dispatches augmentum:session-changed
    }
  }

  // Create the primary renderer + input for the default (legacy) chat surface.
  // These are used by the backward-compat `chat` API object.
  // In multi-surface mode, each ChatSurface creates its own instances.
  _setupPrimarySurface();

  // Multi-model fan-out (passthrough compare) — composer button + popover.
  multiModel.initMultiModel({
    renderMessages,
    getRenderer: () => _primaryRenderer,
  });

  // Wire TTS module's session getter
  tts.setActiveSessionGetter(getActiveSession);
  // Character-voice lookup is injected by narrative/index.js on its own init
  // (avoids a circular import between chat/ and narrative/).

  // Wire code-actions module with session accessor.
  // Code-block edits mutate the active session's tree, so pass the id —
  // a bare sessionStore.save() only writes localStorage stubs and marks
  // nothing dirty for server sync, which silently dropped accepted AI
  // edits on refresh.
  initCodeBlockActions({
    getActiveSession,
    saveSessions: () => sessionStore.save(sessionStore.getActiveId()),
  });

  // Wire project.js (App Builder) bridge. This was never hooked up after
  // the chat.js → chat/ refactor, leaving the module on its no-op stub
  // bridge: background-build delivery into chat silently did nothing and
  // project card mutations (modify/revert/fix) were never persisted.
  setChatBridge({
    getSessions: () => sessionStore.all(),
    getActiveSessionId: () => sessionStore.getActiveId(),
    addChildNode: tree.addChildNode,
    renderMessages,
    saveSessions: () => sessionStore.save(sessionStore.getActiveId()),
  });

  // Global event listeners (shared across all surfaces)
  _initGlobalListeners();

  // ToolCard action wiring + default routing for artifact preview/edit.
  // Do not block first chat surface mount on this cold-path module; cards
  // can wire themselves a tick later without delaying the visible shell.
  import('./tool-card.js')
    .then(({ ensureToolCardActionsWired }) => ensureToolCardActionsWired())
    .catch(() => {});
  document.addEventListener('artifact:preview', async (e) => {
    const id = e.detail?.artifact_id;
    if (!id) return;
    try {
      await _openArtifactPreview(id);
    } catch {
      window.open(`/api/artifacts/${encodeURIComponent(id)}/preview`, '_blank');
    }
  });
  document.addEventListener('artifact:edit', async (e) => {
    const id = e.detail?.artifact_id;
    if (!id) return;
    try { await _openArtifactEditor(id); } catch { /* editor unavailable */ }
  });

  // Language-partner: launch a focused drill from a chat-bubble chip.
  // The partner's `suggest_drill` tool emits a card with a button that
  // dispatches this event. We lazy-import the drill launcher so a chat
  // session that never sees a language partner never pulls in the
  // language-game bundles.
  document.addEventListener('learning:launch_drill', async (e) => {
    const detail = e.detail || {};
    if (!detail.game_id || !detail.lang) return;
    try {
      const { launchDrill } = await import('../learning_games/drill_launcher.js');
      await launchDrill(detail);
    } catch (err) {
      console.warn('[learning] drill launch failed', err);
    }
  });

  // Expose icons globally for other modules
  window.icons = icons;

  // Initialize CodeMind AST engine (non-blocking)
  import('../codemind.js').then(m => m.init()).catch(() => {});

  // Session search
  _initSessionSearch();

  // Mic button (for primary surface input area)
  const primaryInputArea = getPrimaryInputArea();
  if (primaryInputArea) initMicButtonModule(primaryInputArea);

  // Illustrate moment (narrative text selection)
  setIllustrateSessionGetter(getActiveSession);
  setIllustrateModeGetter(() => app.state.mode || 'passthrough');
  initIllustrateMoment();

  // Memory glow
  memoryGlow.memGlowInit(app.state.mode || 'passthrough');
  memoryGlow.memGlowClick();

  // Re-attach build cards for any in-flight app builds (coder-workspace path).
  recoverBuildCards();

  // Chat → Browse bridge: links in chat messages open in the browse reader.
  // Plain left-click only — modifier/middle-click still opens a new tab.
  document.addEventListener('click', (e) => {
    if (e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return;
    const link = e.target.closest('.message-content a.md-link, .message-content a.citation-link');
    if (!link) return;
    const href = link.getAttribute('href');
    if (!href || !/^https?:\/\//i.test(href)) return;
    e.preventDefault();
    document.dispatchEvent(new CustomEvent('augmentum:browse-url', { detail: { url: href } }));
  });

  // Related-files chip strip — click inserts [file:id] token, server-side
  // resolver in proxy/file_token_resolver.py inlines the content before LLM.
  document.getElementById('related-files-strip')?.addEventListener('click', (e) => {
    const chip = e.target.closest('.file-chip');
    if (!chip) return;
    const id = chip.dataset.fileId;
    const name = chip.dataset.name;
    const input = document.getElementById('chat-input');
    if (input) {
      const token = `[file:${id}] ${name}`;
      input.value = input.value ? `${input.value.replace(/\s+$/, '')} ${token}` : token;
      input.focus();
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }
    document.getElementById('related-files-strip')?.classList.add('hidden');
  });

  // Proactive related-files chip is fully disabled across all chat modes.
  // The /api/files/search endpoint runs against an FTS5 index over a
  // 60k+ row file_index table; under contention it has been observed
  // taking 40-80 seconds per call and queueing every other coroutine
  // (auth, polling endpoints, narrative state reads) on the shared
  // aiosqlite worker thread. The feature can be re-enabled per-mode
  // once the search path is faster (or moved off the shared connection).
  const _chatInput = document.getElementById('chat-input');
  if (_chatInput) {
    _chatInput.addEventListener('blur', () => {
      // Defensive: ensure the strip stays hidden if it was ever shown by
      // a leftover render path.
      setTimeout(() => {
        document.getElementById('related-files-strip')?.classList.add('hidden');
      }, 200);
    });
    // Hide once on mount so any pre-rendered chips disappear immediately.
    document.getElementById('related-files-strip')?.classList.add('hidden');
  }

  // New chat button
  document.getElementById('new-chat-btn')?.addEventListener('click', () => createSession());

  // Export/import footer
  _initSessionFooter();

  // Render the sidebar session list for the restored mode
  renderSessionList();

  // Restore the active session's messages so the chat area isn't blank on load.
  // switchSession() does this on user clicks, but initChat never did.
  const restoredId = sessionStore.getActiveId();
  if (restoredId && sessionStore.get(restoredId)) {
    app.state.currentSessionId = restoredId;
    sessionStore.ensureLoaded(restoredId).then(() => renderMessages());
  }

  // --- Window globals for inline onclick handlers in rendered HTML ---

  window.__inspectStoredReasoning = function(btn) {
    const messageEl = btn.closest('.message');
    if (!messageEl) return;
    const template = messageEl.querySelector('.stored-reasoning-data');
    if (!template) return;
    try {
      // <template> parks its inner HTML in a parallel DocumentFragment
      // at .content; reading .textContent on the element itself returns
      // "" because the template has no direct DOM children. Read from
      // the fragment so JSON.parse actually gets the payload.
      const raw = template.content ? template.content.textContent : template.textContent;
      const reasoning = JSON.parse(raw);
      import('../analytical.js').then(m => m.restoreReasoningFromStored(reasoning));
      app.closeImagePanel();
      if (window.innerWidth < 1024) {
        app.closePanel();
        app.openInspectorMobile();
      } else {
        app.state.inspectorVisible = true;
        app.dom.app.setAttribute('data-inspector', 'visible');
      }
    } catch (e) { console.error('Failed to restore reasoning:', e); }
  };

  window.__toggleAgenticStep = function(stepName) {
    if (agenticExpandedSteps.has(stepName)) {
      agenticExpandedSteps.delete(stepName);
    } else {
      agenticExpandedSteps.add(stepName);
    }
    // Re-render by toggling classes in-place
    const contentBox = document.getElementById(`agentic-step-content-${stepName}`);
    if (contentBox) {
      const isExpanded = agenticExpandedSteps.has(stepName);
      contentBox.classList.toggle('expanded', isExpanded);
      contentBox.classList.toggle('collapsed', !isExpanded);
      const block = contentBox.closest('.pipeline-step-block');
      const toggle = block?.querySelector('.pipeline-step-toggle');
      if (toggle) toggle.innerHTML = isExpanded ? icons.chevronDown : icons.chevronRightSmall;
    }
  };

  window._toggleArtifactPreview = function(btn) {
    const artifactId = btn.dataset.previewId;
    const card = btn.closest('.artifact-card');
    if (!card) return;

    // Check if preview already exists
    let preview = card.nextElementSibling;
    if (preview && preview.classList.contains('artifact-preview')) {
      preview.remove();
      btn.classList.remove('active');
      return;
    }

    // Detect format from the card metadata for format-specific rendering
    const metaEl = card.querySelector('.artifact-card-meta');
    const metaText = (metaEl?.textContent || '').toLowerCase();
    const isPdf = metaText.includes('pdf');
    const isSpreadsheet = metaText.includes('xlsx') || metaText.includes('csv');

    // Create preview iframe
    const wrapper = document.createElement('div');
    wrapper.className = 'artifact-preview';
    if (isPdf) wrapper.classList.add('artifact-preview-tall');

    // PDFs need minimal sandbox (browser PDF viewer requires navigation);
    // HTML artifacts need allow-scripts for interactivity
    const sandbox = isPdf ? '' : 'sandbox="allow-scripts allow-forms allow-modals allow-popups"';

    wrapper.innerHTML =
      `<iframe src="/api/artifacts/${encodeURIComponent(artifactId)}/preview" ` +
      `${sandbox} ` +
      `class="artifact-preview-iframe" loading="lazy"></iframe>` +
      `<button class="artifact-preview-close" onclick="this.parentElement.remove()">&times;</button>`;
    card.insertAdjacentElement('afterend', wrapper);
    btn.classList.add('active');

    // For interactive (non-PDF) artifacts, upgrade the iframe to the
    // isolated preview origin when the server has content isolation on.
    // The default same-origin src above gives a null origin where ES
    // modules (CORS-blocked) and localStorage (throws) fail; the
    // isolated origin gives a real foreign origin where both work. Mint
    // is async + best-effort — on 501/disabled the same-origin frame
    // already loaded, so nothing regresses.
    if (!isPdf) {
      const frame = wrapper.querySelector('iframe');
      fetch('/api/content/preview-token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ kind: 'artifact_app', id: artifactId }),
      }).then(r => (r.ok ? r.json() : null)).then((data) => {
        if (!data || !data.token || !data.isolated_origin) return;
        if (!frame.isConnected) return;   // user closed the preview already
        // Different origin ⇒ allow-same-origin is safe (no cookie crossover).
        frame.setAttribute(
          'sandbox',
          'allow-scripts allow-forms allow-modals allow-popups allow-same-origin',
        );
        frame.src = `${data.isolated_origin}/api/artifacts/`
          + `${encodeURIComponent(artifactId)}/preview`
          + `?_pvt=${encodeURIComponent(data.token)}`;
      }).catch(() => {});
    }
  };

  window._openArtifactStudio = function(btn) {
    const id = btn.dataset.editId;
    if (id) import('../studio.js').then(m => m.openStudio(id));
  };

  window._castArtifact = function(btn) {
    const id = btn?.dataset?.castId;
    if (!id) return;
    // Cast the artifact preview URL through the standard picker.
    // /api/artifacts/{id}/preview renders images / docs / generated
    // HTML uniformly so display.web_show@1 handles every supported
    // artifact kind.
    import('../cast-picker.js').then(({ openCastPicker }) => {
      // Pull the on-card name for the picker title (falls back to the
      // artifact id if the metadata hasn't hydrated yet).
      const card = btn.closest('.artifact-card');
      const name = card?.querySelector('.artifact-card-name')?.textContent?.trim();
      openCastPicker({
        anchor: btn,
        capability: 'display.web_show@1',
        content: {
          contentUrl: `/api/artifacts/${encodeURIComponent(id)}/preview`,
          title: name && name !== 'Loading...' ? name : `Artifact ${id.slice(0, 8)}`,
          contentKey: `artifact:${id}`,
          fileId: id,
          metadata: { source: 'chat-artifact-card' },
        },
      });
    }).catch((err) => console.warn('[chat] artifact cast import failed', err));
  };

  window._openPdfAnnotator = function(artifactId) {
    import('../studio.js').then(m => m.openStudio(artifactId, { forceVisualPdf: true }));
  };

  // --- Custom event handlers ---

  // Voice-generated images
  document.addEventListener('augmentum:voice-image', (e) => {
    const { url } = e.detail;
    if (!url) return;
    const session = getActiveSession();
    if (!session) return;
    const parentId = session.activeLeafId || null;
    const node = tree.addChildNode(session, parentId, 'assistant', `![Generated Image](${url})`);
    session.activeLeafId = node.id;
    sessionStore.save(session.id);
    renderMessages();
  });

  // Narrative character chat start
  document.addEventListener('narrative-start-chat', (e) => {
    const { characterId, characterName, greeting, systemPrompt, characterAvatar, userAvatar, characterVoice, lorebook, examples, creatorNotes, groupId, groupMembers, groupMode } = e.detail;
    const id = sessionStore.create('narrative');
    const session = sessionStore.get(id);
    if (!session) return;
    session.title = characterName || 'Character Chat';
    if (characterId) session.characterId = characterId;
    // Group chat fields: persisting groupId is what activates the backend's
    // GroupTurnManager + speaker-aware prompt swap. Without it, group chats
    // degrade to one model cosplaying everyone in a single system blob.
    if (groupId) session.groupId = groupId;
    if (groupMembers) session.groupMembers = groupMembers;
    if (groupMode) session.groupMode = groupMode;
    if (characterAvatar) session.characterAvatar = characterAvatar;
    if (userAvatar) session.userAvatar = userAvatar;
    if (systemPrompt) session.narrativeSystemPrompt = systemPrompt;
    if (lorebook && lorebook.length > 0) {
      session.lorebook = lorebook;
      session.lorebookState = {};
    }
    if (examples) session.narrativeExamples = examples;
    if (creatorNotes) session.narrativeCreatorNotes = creatorNotes;
    if (characterVoice) session.characterVoice = characterVoice;
    if (greeting) {
      const greetingNode = tree.addChildNode(session, null, 'assistant', greeting);
      greetingNode.isGreeting = true;
      session.activeLeafId = greetingNode.id;
    }
    // Mirror the active id into app.state BEFORE setActiveId (which fires
    // augmentum:session-changed synchronously). Listeners read
    // state.currentSessionId, and any other read of it (scene-gen, image,
    // etc.) needs the new value — the previous-session pointer was the
    // root cause of the cross-character chimera bug.
    app.state.currentSessionId = id;
    sessionStore.setActiveId(id);  // dispatches augmentum:session-changed
    sessionStore.save();
    renderSessionList();
    renderMessages();
    // Toast removed — the session list and chat header already reflect
    // the new chat, so the toast was duplicative confirmation noise.
  });
}

/**
 * Set up the primary surface's renderer and input.
 * Mounts into the existing DOM elements (#chat-scroll, .input-area)
 * to maintain backward compatibility.
 */
function _setupPrimarySurface() {
  // The primary surface wraps the existing singleton DOM elements
  // until the full migration to surface-created DOM is complete.
  // This ensures the legacy chat experience works identically.

  const chatScroll = document.getElementById('chat-scroll');
  const chatMessages = document.getElementById('chat-messages');
  const inputArea = getPrimaryInputArea();

  if (chatScroll && chatMessages) {
    _primaryRenderer = new MessageRenderer({
      mode: app.state.mode || 'passthrough',
      onAction: _handlePrimaryAction,
      highlightHooks: _getHighlightHooks(),
    });
    // Instead of createDOM, adopt the existing elements
    _primaryRenderer.scrollEl = chatScroll;
    _primaryRenderer.messagesEl = chatMessages;
    _primaryRenderer.emptyStateEl = document.getElementById('empty-state');
    _primaryRenderer._initScrollTracking();
    _primaryRenderer._initDelegatedActions();

    // Mount the offer feed on the same container. Offers (chat-LLM
    // Install/Save/Switch chips) are pushed live via the notification
    // WS and backfilled from /api/notify/feed on attach.
    // Spec: docs/superpowers/specs/2026-06-02-offer-substrate-design.md
    try {
      if (!_primaryRenderer._offerFeedHandle) {
        // threadId is a getter, not a value: the feed mounts once but the
        // active session changes under it, so an id captured here would pin
        // the filter to whichever chat happened to be open first — which is
        // how offers leaked into every chat on every device.
        _primaryRenderer._offerFeedHandle = mountOfferFeed(chatMessages, {
          threadId: () => sessionStore.getActiveId() || '',
          onAfterAction: _onOfferAction,
        });
      }
    } catch { /* offers are best-effort surface; no fallback needed */ }
  }

  // Primary input: use existing input area (don't create new DOM)
  // The existing event handlers in app.js handle the primary input.
  // In multi-surface mode, each ChatSurface creates its own ChatInput.
}

// A gated-tool offer that finishes synchronously (image_generation) returns a
// `deliverable` on Accept. Append it into the originating session's last
// assistant message — persisted in the tree exactly like the inline tool path,
// so it renders inline AND survives refresh. buildMessagesForAPI redacts the
// img markdown before it reaches the model, so this never leaks into context.
// Detached gated tools (builds/ebooks) carry no deliverable — they land in
// their own home (project card / library), so there's nothing to deliver here.
function _onOfferAction(_offerId, actionId, result) {
  if (actionId !== 'accept' || !result) return;
  const d = result.deliverable;
  if (!d || d.kind !== 'image' || !d.url) return;
  // Target the originating session by id; fall back to the active one (the
  // chip is rendered in the active session's container, so they usually match).
  const session = (d.session_id && sessionStore.get(d.session_id)) || getActiveSession();
  if (!session || !session.tree) return;
  const path = tree.getActivePath(session);
  let target = null;
  for (let i = path.length - 1; i >= 0; i--) {
    if (path[i].role === 'assistant') { target = path[i]; break; }
  }
  if (!target) return;
  if (typeof target.content === 'string' && target.content.includes(d.url)) return;
  target.content = (target.content || '') + `\n\n![Generated Image](${d.url})`;
  sessionStore.save(session.id);
  if (session.id === sessionStore.getActiveId()) renderMessages();
}

function _getHighlightHooks() {
  return {
    mermaid: (container) => {
      if (typeof mermaid === 'undefined') return;
      const blocks = (container || document).querySelectorAll('.mermaid-block:not(.mermaid-rendered)');
      blocks.forEach(async (block) => {
        block.classList.add('mermaid-rendered');
        const code = decodeURIComponent(block.dataset.mermaid);
        try {
          const id = 'mermaid-' + Math.random().toString(36).substring(2, 10);
          const { svg } = await mermaid.render(id, code);
          block.innerHTML = svg;
          block.classList.add('mermaid-success');
        } catch {
          block.classList.add('mermaid-error');
        }
      });
    },
    artifactCards: (container) => {
      const _iconMap = {
        pdf: icons.filePdf, docx: icons.fileDocx, pptx: icons.filePptx,
        xlsx: icons.fileXlsx, png: icons.fileImage, default: icons.fileDefault
      };
      const cards = (container || document).querySelectorAll('.artifact-card:not(.artifact-loaded)');
      cards.forEach(async (card) => {
        card.classList.add('artifact-loaded');
        const artifactId = card.dataset.artifactId;
        try {
          const resp = await fetch(`/api/artifacts/${artifactId}`);
          if (!resp.ok) return;
          const info = await resp.json();
          const icon = _iconMap[info.format] || _iconMap.default;
          const sizeKB = info.size_bytes ? `${(info.size_bytes / 1024).toFixed(1)} KB` : '';
          const fmt = (info.format || '').toUpperCase();
          card.querySelector('.artifact-card-icon').innerHTML = icon;
          card.querySelector('.artifact-card-name').textContent = info.display_name || info.filename || 'Artifact';
          card.querySelector('.artifact-card-meta').textContent = [fmt, sizeKB].filter(Boolean).join(' \u2022 ');
        } catch { /* stays in loading state */ }
      });
    },
  };
}

async function _handlePrimaryAction(action, nodeId, data) {
  const session = getActiveSession();
  if (!session && action !== 'send' && action !== 'lightbox') return;

  switch (action) {
    case 'branch':
      tree.switchToSibling(session, nodeId, data);
      renderMessages();
      sessionStore.save(session.id);
      break;

    case 'regenerate': {
      const node = session.tree[nodeId];
      if (!node || node.role !== 'assistant') break;
      session.activeLeafId = node.parentId;
      sessionStore.save(session.id);
      renderMessages();
      app.state.isStreaming = true;
      _primaryStreamResponse(session);
      break;
    }

    case 'continue':
      if (session) {
        // Extend the trailing assistant message in place — no fake
        // [Continue] user node, no new assistant node. The backend
        // tells the provider to continue the partial verbatim (DeepSeek
        // prefix, Anthropic native, llama-server add_generation_prompt
        // false, synthetic-user fallback).
        const lastNode = session.tree?.[session.activeLeafId];
        if (lastNode && lastNode.role === 'assistant') {
          // Clear the interrupted marker so the UI doesn't keep showing
          // the incomplete badge while the continuation streams in.
          if (lastNode.interrupted) {
            delete lastNode.interrupted;
            delete lastNode.error_message;
            delete lastNode.error_kind;
          }
          sessionStore.save(session.id);
          app.state.isStreaming = true;
          _primaryStreamResponse(session, { continueLastAssistant: true });
        }
      }
      break;

    case 'impersonate':
      if (session) _primaryImpersonate(session);
      break;

    case 'edit': {
      // Shared inline-edit UI (renderer-owned so it works in every surface).
      const node = session.tree[nodeId];
      if (!node) break;
      _primaryRenderer?.beginInlineEdit(nodeId, node.content);
      break;
    }

    case 'save-edit': {
      const node = session.tree[nodeId];
      if (!node) break;
      // Remove edit UI
      const msgEl = _primaryRenderer?.messagesEl?.querySelector(`[data-node-id="${nodeId}"]`);
      const editWrap = msgEl?.querySelector('.edit-wrap');
      if (editWrap) editWrap.remove();

      if (node.role === 'assistant') {
        node.content = data.content;
        sessionStore.save(session.id);
        renderMessages();
      } else {
        // User edit → create branch and re-send
        const parentId = node.parentId;
        const newUserNode = tree.addChildNode(session, parentId, 'user', data.content);
        session.activeLeafId = newUserNode.id;
        sessionStore.save(session.id);
        renderMessages();
        app.state.isStreaming = true;
        _primaryStreamResponse(session);
      }
      break;
    }

    case 'cancel-edit': {
      _primaryRenderer?.cancelInlineEdit();
      break;
    }

    case 'delete': {
      const descendants = tree.countDescendants(session, nodeId);
      if (descendants > 0 || Object.keys(session.tree).length > 5) {
        if (!confirm(`Delete this message${descendants > 0 ? ` and ${descendants} replies` : ''}?`)) return;
      }
      const node = session.tree[nodeId];
      const parentId = node?.parentId;
      tree.removeNodeAndDescendants(session, nodeId);
      if (parentId && session.tree[parentId]) {
        session.activeLeafId = tree.getDeepestLeaf(session, parentId);
      } else {
        // Parent gone (root deletion or orphan) — find any surviving leaf
        session.activeLeafId = tree.findAnyLeaf(session);
      }
      sessionStore.save(session.id);
      renderMessages();
      break;
    }

    case 'tts': {
      const node = nodeId && session ? session.tree?.[nodeId] : null;
      tts.ttsPlayMessage(data?.text, data?.button, { speakerName: node?.speakerName || '' });
      break;
    }

    case 'vote':
      if (data?.balancerId && data?.vote) {
        fetch(`/api/balancers/${data.balancerId}/vote`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ vote: data.vote, model: data.model, backend: data.backend }),
        }).catch(() => {});
        if (session) {
          const node = session.tree[nodeId];
          if (node) { node.ab_vote = data.vote; sessionStore.save(session.id); }
        }
      }
      break;

    case 'code': {
      const codeAction = data?.action;
      const codeHeader = data?.codeHeader;
      if (!codeAction || !codeHeader) break;
      switch (codeAction) {
        case 'preview-code':    toggleHtmlPreview(codeHeader); break;
        case 'preview-svg':     toggleSvgPreview(codeHeader); break;
        case 'run-code':        runCodeBlock(codeHeader); break;
        case 'auto-fix':        autoFixCodeBlock(codeHeader); break;
        case 'ask-ai-edit':     showAskAiPrompt(codeHeader); break;
        case 'edit-code':       toggleCodeEdit(codeHeader); break;
        case 'quick-actions':   showQuickActionsMenu(codeHeader); break;
        case 'download-code':   downloadCodeBlock(codeHeader); break;
        case 'save-to-library': saveCodeBlockToLibrary(codeHeader); break;
        default:
          document.dispatchEvent(new CustomEvent('chat:action', { detail: { action, nodeId, data } }));
      }
      break;
    }

    case 'lightbox':
      if (data?.src) openImageLightbox(data.src);
      break;

    case 'open-in-browse':
      // Knowledge-pack source clicked. Lazy-import browse so users who
      // never install a pack don't pay the load cost. Failure mode: if
      // browse.js can't load (offline build, removed module), we fall
      // back to a toast — better than a silent click. The Browse panel
      // owns history/tab state from there.
      if (data?.url) {
        import('../browse.js')
          .then(m => m.openInBrowse?.(data.url))
          .catch(err => {
            console.warn('failed to open browse for pack source', err);
          });
      }
      break;

    case 'render-youtube':
      // Rehydrate inline search cards on reload. Direct-mode videos are
      // deliberately NOT auto-reopened — popping the YouTube panel on
      // every refresh for every direct-mode node would be disruptive.
      if (data?.data?.youtube_mode === 'search' && data.messageEl) {
        renderYouTubeCards(data.data, data.messageEl);
      }
      break;

    case 'render-project-card':
      // The renderer emits this for every assistant node with a
      // .projectArtifact, both on live build completion and on page
      // reload. The pre-refactor chat.js called _renderProjectCard
      // directly; the modular split moved to an action but never
      // wired a handler — which is why completed app-builder cards
      // stopped appearing in chat until this was restored.
      if (data?.node && data.contentEl) {
        renderProjectCard(data.node, data.contentEl);
      }
      break;

    case 'graduate-to-becca': {
      // Narrative → Becca memory graduation (Lane 3 §4.6). Single content-
      // crossing path; runtime labeler processes the graduated content.
      // Silently no-ops if companion_runtime_enabled is off (503).
      const content = (data?.content || '').trim();
      if (!content) break;
      const btn = data?.button;
      const userId = window.augmentumState?.currentUserId
        || (await fetch('/api/auth/me', { credentials: 'same-origin' })
          .then(r => r.ok ? r.json() : null)
          .then(j => j?.id || j?.user_id || '')
          .catch(() => ''));
      if (!userId) {
        if (btn) {
          const orig = btn.innerHTML;
          btn.innerHTML = '!';
          setTimeout(() => { btn.innerHTML = orig; }, 1500);
        }
        break;
      }
      fetch('/api/companion/narrative/graduate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({
          user_id: userId,
          content,
          source_session_id: session?.id || '',
          importance: 0.6,
        }),
      }).then(r => r.json()).then((j) => {
        if (btn) {
          const ok = j && j.ok;
          const orig = btn.innerHTML;
          btn.innerHTML = ok ? '✓' : '!';
          setTimeout(() => { btn.innerHTML = orig; }, 1500);
        }
      }).catch(() => {
        if (btn) {
          const orig = btn.innerHTML;
          btn.innerHTML = '!';
          setTimeout(() => { btn.innerHTML = orig; }, 1500);
        }
      });
      break;
    }

    case 'rerun-as': {
      // DPO training signal — user picked a different mode from the inline
      // mode picker (renderer._openModePickerPopover). We record the
      // (original_winner, chosen_target) pair so the dispatcher learns
      // the preference. Lane 3 §9.
      const current = session?.mode || 'passthrough';
      const chosen = data?.targetMode;
      if (!chosen || chosen === current) break;
      const btn = data?.button;
      const content = (data?.content || '').trim();
      const userId = window.augmentumState?.currentUserId
        || (await fetch('/api/auth/me', { credentials: 'same-origin' })
          .then(r => r.ok ? r.json() : null)
          .then(j => j?.id || j?.user_id || '')
          .catch(() => ''));
      if (!userId) break;
      fetch('/api/companion/rerun_as', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({
          user_id: userId,
          intent_text: content.slice(0, 4000),
          original_winner: current,
          chosen_target: chosen,
        }),
      }).then(r => r.json()).then((j) => {
        // Best-effort visual receipt on the chosen mode row so the user
        // sees the request landed. The picker popover has already closed,
        // so we toast instead of trying to flash the (gone) button.
        const ok = j && j.ok;
        showToast(ok ? `Will rerun in ${chosen} next time.` : "Couldn't save that preference.", ok ? 'success' : 'error');
        if (btn && btn.isConnected) {
          const orig = btn.innerHTML;
          btn.innerHTML = ok ? '✓' : '!';
          setTimeout(() => { btn.innerHTML = orig; }, 1500);
        }
      }).catch(() => {});
      break;
    }

    case 'send':
      if (data?.text) {
        document.dispatchEvent(new CustomEvent('augmentum:send', { detail: { text: data.text } }));
      }
      break;

    default:
      // Unknown action — dispatch as event for any listener
      document.dispatchEvent(new CustomEvent('chat:action', { detail: { action, nodeId, data } }));
  }
}

// ---------------------------------------------------------------------------
// Primary Surface Send Pipeline
// ---------------------------------------------------------------------------

async function _primarySend(text, images, docs) {
  if (app.state.isStreaming) return;

  // Cold-start guard. If the model catalog has loaded and is genuinely empty,
  // a real send would fail with an opaque backend error ("Something went
  // sideways on the model"). Surface a clear, actionable message and route the
  // user into provider setup instead. Only fires when we're certain —
  // getModelsSync() returns [] (loaded + empty); a null (not-yet-loaded) list
  // falls through unguarded, so a cold cache never blocks a legitimate send.
  const hasContent = !!(text || (images && images.length) || (docs && docs.length));
  if (hasContent) {
    const models = getModelsSync();
    if (Array.isArray(models) && models.length === 0) {
      showToast('No models set up yet — connect a provider to start chatting.', 'error', 6000, {
        action: { label: 'Set up', onClick: () => document.getElementById('model-selector')?.click() },
      });
      return;
    }
  }

  if (!text && (!images || images.length === 0) && (!docs || docs.length === 0)) {
    // Empty send = re-send (used by regenerate/continue which already set up the tree)
    const session = getActiveSession();
    if (session) {
      app.state.isStreaming = true;
      if (await _maybeRunFanout(session)) return;
      await _primaryStreamResponse(session);
    }
    return;
  }

  app.state.isStreaming = true;

  // Create session if needed
  if (!sessionStore.getActiveId()) {
    createSession();
  }

  const session = getActiveSession();
  if (!session) {
    app.state.isStreaming = false;
    return;
  }

  // Add user node
  const parentId = session.activeLeafId || null;
  const userNode = tree.addChildNode(session, parentId, 'user', text);
  if (images && images.length > 0) userNode.images = images;
  session.activeLeafId = userNode.id;
  _maybeGenerateTitle(session);
  sessionStore.save(session.id);

  // Render the new user message by APPENDING it — not a full-thread
  // rebuild. renderMessages() re-parses every prior message's markdown
  // synchronously (O(all messages)), which froze the page for seconds on
  // long threads right after hitting send. appendMessage adds just this
  // node. (Branch/edit/delete still use renderMessages — the path changes.)
  if (_primaryRenderer) {
    _primaryRenderer.appendMessage(userNode, session);
  } else {
    renderMessages();
  }

  if ((app.state.mode || session.mode || '') === 'agentic') {
    showAgenticTaskStarting(text);
  }

  // Memory glow — recalling state
  memoryGlow.memGlowRecalling();

  // Multi-model fan-out — one user turn, N model responses as siblings.
  if (await _maybeRunFanout(session)) return;

  // Stream response
  await _primaryStreamResponse(session);
}

/**
 * Route the turn through the multi-model compare path when armed.
 * Returns true when the fan-out ran (it owns isStreaming + finalize);
 * false falls through to the normal single-stream path.
 */
async function _maybeRunFanout(session) {
  const mode = app.state.mode || session.mode || 'passthrough';
  if (mode !== 'passthrough' || !multiModel.isFanoutActive()) return false;
  const tools = (app.state.passthroughTools || []).join(',');
  try {
    return await multiModel.runFanout(session, { tools });
  } catch (err) {
    // Fan-out must never strand the composer — fall back to single-stream.
    console.warn('[multi-model] fan-out failed, falling back', err);
    return false;
  }
}

async function _primaryImpersonate(session) {
  if (app.state.isStreaming) return;
  const mode = app.state.mode || session.mode || 'passthrough';
  const persona = await fetchActivePersona();
  const input = app.dom.chatInput;
  await runImpersonate(session, {
    mode,
    sessionId: session.id || '',
    userName: persona?.name || 'User',
    charName: session.characterName || session.title || 'the character',
    personaDesc: persona?.description || '',
    onStart: () => {
      app.state.isStreaming = true;
      if (input) {
        input.value = '';
        input.dispatchEvent(new Event('input', { bubbles: true }));
      }
    },
    onText: (acc) => {
      if (!input) return;
      input.value = acc;
      input.dispatchEvent(new Event('input', { bubbles: true }));
    },
    onEnd: () => {
      app.state.isStreaming = false;
      if (input) input.focus();
    },
  });
}

async function _primaryStreamResponse(session, opts = {}) {
  if (!_primaryRenderer) return;
  _pendingYouTubeMeta = null;
  _pendingAbTest = null;
  _pendingAgenticArtifactCards = [];

  // Continue button: resume streaming into the existing trailing
  // assistant message instead of creating a fresh one.
  const continueLastAssistant = !!opts.continueLastAssistant;
  // Fresh user turn — reset the per-turn auto-resume budget. Continues
  // (manual OR auto) deliberately don't reset so a single user message
  // can't burn through unlimited retries.
  if (!continueLastAssistant) _autoResumeAttempted = false;
  if (continueLastAssistant) {
    const resumed = _primaryRenderer.resumeStreamingMessage(session.activeLeafId);
    if (!resumed) {
      // Fall through to a normal new-streaming-message if we couldn't
      // find the assistant DOM (shouldn't happen if the button was
      // visible, but defensive — the user clicked "continue" and
      // expects *something* to happen).
      _primaryRenderer.createStreamingMessage(session);
    }
  } else {
    _primaryRenderer.createStreamingMessage(session);
  }
  _primaryRenderer.scrollToBottom(false, true);

  // Abort any existing stream
  if (_activeStream) _activeStream.abort();

  // Create a ChatStream for this request
  const stream = new ChatStream({
    onContent: (text) => {
      _primaryRenderer.appendToStreaming(text);
      // First content delta is the implicit signal that prefill ended
      // (backend doesn't emit stage_complete for prefill — see comment
      // in llama_cpp.py at the prefill stage start). Stop polling the
      // progress endpoint so we're not pinging it 2x/sec for nothing.
      import('./prefill-progress.js').then(m => m.stopPrefillPolling());
      // Progressive TTS feed
      const s = getSettings();
      if (s.voiceAutoRead) {
        tts.ttsProgressiveFeed(text);
      }
    },
    onMeta: (meta) => _handlePrimaryMeta(meta),
    onComplete: (result) => _handlePrimaryStreamComplete(session, result, { continueLastAssistant }),
    onError: (err) => {
      app.state.isStreaming = false;
      _pendingYouTubeMeta = null;
      _pendingAbTest = null;
      _pendingAgenticArtifactCards = [];

      // Persist whatever was streamed so far as an assistant node, so
      // (a) the user can copy/refer to the partial answer, (b) the
      // existing regenerate button on the message bubble works as a
      // retry affordance — clicking it resets activeLeafId to the
      // parent (user's message) and re-streams a fresh response with
      // identical context. Without persistence, error responses vanish
      // on the next render and the user has to re-type their question.
      //
      // For continuation: the streamingEl IS the existing assistant
      // node (resumeStreamingMessage swapped it in). The renderer
      // already has the partial+streamed text in dataset.rawContent —
      // just write it back to the existing node, don't branch.
      const rawContent = _primaryRenderer.getStreamingRawContent() || '';
      if (continueLastAssistant) {
        const existingNode = session.tree?.[session.activeLeafId];
        if (existingNode && existingNode.role === 'assistant') {
          existingNode.content = rawContent;
          existingNode.interrupted = true;
          existingNode.error_message = err.message || 'connection failed';
          if (err.kind) existingNode.error_kind = err.kind;
          sessionStore.syncNow(session.id);
        }
      } else if (rawContent.trim()) {
        const parentId = session.activeLeafId;
        const assistantNode = tree.addChildNode(session, parentId, 'assistant', rawContent);
        assistantNode.interrupted = true;
        assistantNode.error_message = err.message || 'connection failed';
        if (err.kind) assistantNode.error_kind = err.kind;
        session.activeLeafId = assistantNode.id;
        _primaryRenderer.setStreamingNodeId(assistantNode.id);
        sessionStore.syncNow(session.id);
      }

      // One-shot auto-resume on network drop. Conditions:
      //   - fetch network failure (TypeError, not a structured backend error)
      //   - we haven't already burned this turn's retry
      //   - we have a partial worth resuming
      //   - we're online (the navigator hint isn't authoritative but it's a
      //     cheap pre-check before we blow 2s on a probe)
      // The flow: keep the interrupted node, show "Reconnecting…", probe
      // /api/version, then continue from the partial OR fall through to
      // normal toast+finalize if the probe fails.
      const isNetworkErr = !err?.isBackendError && err?.name === 'TypeError';
      const canAutoResume = isNetworkErr
        && !_autoResumeAttempted
        && rawContent.trim().length > 0
        && (typeof navigator === 'undefined' || navigator.onLine !== false);

      if (canAutoResume) {
        _autoResumeAttempted = true;
        _primaryRenderer.setStallBanner('thinking');
        tts.ttsProgressiveCancel();
        setTimeout(async () => {
          let alive = false;
          try {
            const probe = await fetch('/api/version', { method: 'GET', cache: 'no-store' });
            alive = probe.ok;
          } catch { /* network still down */ }
          if (alive) {
            // Backend reachable — clear interrupted state on the partial
            // and fire continue. The renderer will resume the existing
            // bubble; the stall banner clears on first content delta.
            const node = session.tree?.[session.activeLeafId];
            if (node && node.role === 'assistant') {
              delete node.interrupted;
              delete node.error_message;
              delete node.error_kind;
            }
            sessionStore.save(session.id);
            app.state.isStreaming = true;
            _primaryStreamResponse(session, { continueLastAssistant: true });
          } else {
            // Probe failed — fall through to the normal finalize+toast path.
            _primaryRenderer.setStallBanner('none');
            _primaryRenderer.finalizeStreaming(session, false);
            showToast("Couldn't reconnect — your last reply may be partial.", 'error');
          }
        }, 2000);
        return;
      }

      // `complete=false` renders the structural interrupted badge and
      // attaches the standard message-actions bar (which includes the
      // regenerate button — that's the retry path).
      _primaryRenderer.finalizeStreaming(session, false);
      tts.ttsProgressiveCancel();
      // Differentiate backend (server-side) errors from client/network
      // errors so the toast wording matches what actually went wrong.
      // Softer copy than "Model backend error" / "Response interrupted"
      // — both of those read as crash reports; the user just wants to
      // know what to do next.
      const isBackend = err?.isBackendError;
      const msg = err.message || 'connection lost';
      const text = isBackend
        ? `Something went sideways on the model — ${msg}`
        : `Reply didn't finish — ${msg}`;
      showToast(text, 'error');
    },
  });

  const mode = app.state.mode || session.mode || 'passthrough';
  const tools = mode === 'narrative' ? 'none' : (app.state.passthroughTools || []).join(',');

  // Initialize progressive TTS pipeline if auto-read is enabled.
  // Lost in the chat.js decomposition (commit c497173); without this,
  // ttsProgressiveFeed() no-ops on every token because the pipeline is null.
  const _ttsSettings = getSettings();
  if (_ttsSettings.voiceAutoRead) {
    tts.ttsProgressiveCancel();
    const _isNarrative = session && session.mode === 'narrative';
    const _voice = (_isNarrative && session.characterVoice)
      ? session.characterVoice
      : (_ttsSettings.voiceDefaultVoice || '');
    const _chunkMode = _ttsSettings.voiceTtsChunking || 'sentence';
    const _speed = _ttsSettings.voiceSpeed || 1.0;
    const _buffer = new tts.TtsSentenceBuffer(_chunkMode);
    const _pipeline = new tts.TtsAudioPipeline(_voice, _speed, _isNarrative, null);
    tts._installActivePipeline(_buffer, _pipeline);
    const _arBtn = document.getElementById('auto-read-btn');
    if (_arBtn) _arBtn.classList.add('tts-streaming');
  }

  _activeStream = stream;
  const selectedModel = (app.state.currentModel || '').trim();
  if (selectedModel && selectedModel !== 'default') {
    addToRecentModels(selectedModel);
  }
  await stream.send(session, {
    model: selectedModel,
    mode,
    tools,
    continueLastAssistant,
  });
  _activeStream = null;
}

// Stall-banner state. The stream watchdog emits stalled/thinking
// independently; the banner shows at most one. 'stalled' wins because
// it's the actionable, alarming variant — 'thinking' is just a soft hint.
const _stallBannerState = { thinking: false, stalled: false };
function _refreshStallBanner() {
  if (!_primaryRenderer) return;
  if (_stallBannerState.stalled) _primaryRenderer.setStallBanner('stalled');
  else if (_stallBannerState.thinking) _primaryRenderer.setStallBanner('thinking');
  else _primaryRenderer.setStallBanner('none');
}

function _handlePrimaryMeta(meta) {
  if (!_primaryRenderer) return;

  // Content-based stall watchdog from ChatStream. Two signals:
  //  - stalled: 15s without content (alarming; pulses send-btn, shows
  //    inline banner with Abort & retry)
  //  - thinking: 4s without content (soft hint, inline only)
  // Heartbeats deliberately do NOT reset these (see stream.js).
  if (typeof meta.stalled === 'boolean') {
    const btn = document.getElementById('send-btn');
    if (btn) btn.classList.toggle('is-stalled', meta.stalled);
    _stallBannerState.stalled = meta.stalled;
    _refreshStallBanner();
  }
  if (typeof meta.thinking === 'boolean') {
    _stallBannerState.thinking = meta.thinking;
    _refreshStallBanner();
  }
  // Heartbeat carries no content but keeps the renderer's "thinking" UI alive.
  if (meta.heartbeat) {
    if (meta.phase) _primaryRenderer.updateStreamingStatus(meta.phase);
    return;
  }

  // Narrative internal-tool activity (recall + lorebook) — route the rich
  // meta into the live activity strip instead of dropping all but the raw
  // status string. These are the loop's own status tags (recall_loop.py).
  if (meta.status === 'narrative_tool_used') {
    _primaryRenderer.addNarrativeActivity({
      tool: meta.tool,
      args: meta.args,
      resultPreview: meta.result_preview,
      kind: 'tool',
    });
    return;
  }
  if (meta.status === 'tool_preamble_suppressed') {
    return;  // internal — nothing user-facing, already dropped from prose
  }
  if (meta.status === 'recall_tool_loop_synthesizing') {
    _primaryRenderer.updateStreamingStatus('Composing the reply…');
    return;
  }
  if (meta.status) _primaryRenderer.updateStreamingStatus(meta.status);
  if (meta.phases) {
    _primaryRenderer.updateStreamThinking(meta.phases, meta.complexity);
    // Also update inspector
    import('../analytical.js').then(m => {
      m.renderReasoningPhases(meta.phases, meta.complexity);
    }).catch(() => {});
  }
  if (meta.phase_content_delta && meta.phase) {
    _primaryRenderer.addStreamPhaseContent(meta.phase, meta.phase_content_delta);
    import('../analytical.js').then(m => {
      m.addPhaseContentDelta(meta.phase, meta.phase_content_delta);
    }).catch(() => {});
  }
  if (meta.tool_call) {
    _primaryRenderer.addStreamToolCall(meta.tool_call);
    // Only draw the legacy pill when the unified event stream isn't active —
    // tool_start / tool_complete handles presentation once we've seen tool_start.
    if (meta.tool_call.phase === 'passthrough' && !_unifiedToolActive) {
      _primaryRenderer.renderToolCallResult(meta.tool_call);
    }
    // Side-effect deliverables (open YouTube panel, image card below
    // the message, etc.) — tool_call fires once per completion even
    // when the unified event path is active, so this is the single
    // dispatch point that works for both event shapes.
    if (meta.tool_call.success !== false && meta.tool_call.result_metadata) {
      if (meta.tool_call.tool === 'youtube') {
        _pendingYouTubeMeta = meta.tool_call.result_metadata;
      }
      renderToolDeliverable(
        meta.tool_call.tool,
        meta.tool_call.result_metadata,
        { messageEl: _primaryRenderer.streamingEl, toolCard: null },
      );
    }
    // Final app-builder payload rides on tool_call.project. Kept for
    // parity with the legacy monolithic handler — the modular refactor
    // dropped these two lines, which is why the build monitor only
    // appeared on refresh (recoverBuildMonitor) instead of live.
    if (meta.tool_call.project) {
      handleProjectResult(meta.tool_call.project);
    }
  }
  // Build mode kicks off a build and hands the conversation a build_id. The
  // build card subscribes to the shared build stream and tracks it live — no
  // obstructive top bar, works from any mode.
  //
  // Two events can carry the build_id: the new `build_started` (direct/gated
  // paths) and the legacy `project_progress` (the agentic tool-chain path).
  // Mount the SAME card from either, so the tab that STARTED the build shows
  // progress live instead of only appearing after a refresh (when
  // recoverBuildCards re-attaches it). handleBuildStarted is idempotent, so
  // repeated events for one build_id mount once.
  if (meta.build_started?.build_id) {
    handleBuildStarted(meta.build_started);
  }
  if (meta.project_progress?.build_id) {
    handleBuildStarted({
      build_id: meta.project_progress.build_id,
      name: meta.project_progress.name,
    });
  }

  // Load-balancer ab_test strategy emits which model/backend served
  // this response. Stashed here and attached to the finalized node at
  // stream-complete so renderer.js renders the 👍/👎 vote row.
  if (meta.ab_test) {
    _pendingAbTest = meta.ab_test;
  }

  // Adaptive multi-tool chain — "Multi-step · N steps" label above
  // the streaming message while the planner works through a sequence
  // of tool calls. Chain state is held at module scope so chain_step
  // handlers below can reference the announced total.
  if (meta.chain) {
    _streamChainState = meta.chain;
    const status = meta.chain.status;
    if (status === 'planning') {
      _primaryRenderer.updateStreamingStatus('planning');
    } else if (status === 'synthesizing') {
      _primaryRenderer.updateStreamingStatus('synthesizing');
    } else if (status === 'running' || meta.chain.total_steps) {
      _primaryRenderer.showChainPlanIndicator(meta.chain);
    }
  }

  // Per-step progress inside an adaptive chain. Running state surfaces
  // as a "Using X" indicator + "Step n/total" status; terminal state
  // draws a legacy tool pill (renderToolCallResult) so the user sees
  // a checkmark/cross per step. Guarded by !_unifiedToolActive to
  // avoid double-drawing when the same turn also emits tool_start /
  // tool_complete events.
  if (meta.chain_step) {
    const step = meta.chain_step;
    if (step.status === 'running') {
      if (!_unifiedToolActive) _primaryRenderer.chainToolRunning(step.tool);
      const total = _streamChainState?.total_steps || '?';
      _primaryRenderer.updateStreamingStatus(`Step ${step.id}/${total}`);
    } else if ((step.status === 'done' || step.status === 'error' || step.status === 'failed') && !_unifiedToolActive) {
      _primaryRenderer.chainToolDone(step.tool, step.status === 'done');
    }
  }
  if (meta.tool_status === 'running' && meta.tool_names && !_unifiedToolActive) {
    _primaryRenderer.showToolIndicator(meta.tool_names);
  }
  // Unified tool events — replace bulk "Using..." pill with per-tool cards.
  if (meta.tool_start) {
    _unifiedToolActive = true;
    _primaryRenderer.renderToolStart(meta.tool_start);
  }
  if (meta.tool_progress) {
    _primaryRenderer.renderToolProgress(meta.tool_progress);
  }
  if (meta.tool_complete) {
    _primaryRenderer.renderToolComplete(meta.tool_complete);
  }

  // Companion (becca_direct) mid-stream tool dispatch. The backend
  // streams an announce/result pair per <tool:NAME /> tag Becca emits;
  // presentation reuses the unified tool cards, and each ui_effect is
  // routed through the intent-action router — the same channels the
  // voice path uses (open surface, start playback, candidate cards).
  // Before this block these chunks were silently dropped, so her tool
  // calls were invisible and their surface effects never fired.
  if (meta.becca_tool_call) {
    _unifiedToolActive = true;
    const id = `becca-${++_beccaToolSeq}`;
    _beccaOpenToolIds.push(id);
    _primaryRenderer.renderToolStart({
      id,
      tool: meta.becca_tool_call.tool,
      context: _beccaToolContext(meta.becca_tool_call.args),
    });
    // Long-horizon progress: image generation holds the turn through
    // model load + diffusion (up to minutes). The queue already
    // exposes stage + step progress at /api/image/generation-status
    // (Studio/voice poll it) — mirror it into this tool card so her
    // chat doesn't sit on an indeterminate chip.
    if (meta.becca_tool_call.tool === 'image_generation'
        || meta.becca_tool_call.tool === 'image.generate_with_defaults') {
      _startBeccaImagePoll(id);
    }
  }
  if (meta.becca_tool_result) {
    const r = meta.becca_tool_result;
    const id = _beccaOpenToolIds.shift();
    _stopBeccaImagePoll();
    if (id) {
      _primaryRenderer.renderToolComplete({
        id,
        tool: r.tool,
        success: r.ok !== false,
        duration_ms: r.duration_ms,
        error: r.error || undefined,
      });
    }
    const effects = Array.isArray(r.ui_effects) ? r.ui_effects : [];
    if (r.ok !== false && effects.length) {
      import('../intent-action-router.js').then(m => {
        for (const fx of effects) {
          if (!fx || !fx.kind) continue;
          m.dispatchIntentAction({
            action: r.tool,
            surface: { channel: fx.kind, payload: fx.payload || {} },
          });
        }
      }).catch(err => console.warn('[becca] ui_effect routing failed', err));
    }
  }
  if (meta.becca_tool_budget_exhausted) {
    _primaryRenderer.updateStreamingStatus('tool budget reached');
  }
  if (meta.becca_handoff) {
    // Channel handoff (coder / narrative / agentic) terminates the turn;
    // route through navigate.open_surface so the target surface mounts.
    const channel = meta.becca_handoff.channel || '';
    if (channel) {
      import('../intent-action-router.js').then(m => {
        m.dispatchIntentAction({
          action: 'becca.handoff',
          surface: { channel: 'navigate.open_surface', payload: { surface: channel } },
        });
      }).catch(() => {});
    }
  }

  // Backend stage events — model_load / model_swap / slot_restore /
  // prefill. Updates the streaming status label with the stage's
  // human-readable label + detail, replacing the previous opaque
  // "Loading model…" with "Loading model · deepseek-v3-instruct".
  // The legacy ``meta.status`` (string) handler above still fires for
  // backwards compat, but stage_start is richer and wins on render
  // order because it runs after the .status path.
  if (meta.stage_start) {
    const s = meta.stage_start;
    const label = s.label || s.stage || '';
    const text = s.detail ? `${label} · ${s.detail}` : label;
    if (text) _primaryRenderer.updateStreamingStatus(text);
    // Long-context prefills (30-180s) deserve a live progress bar.
    // The backend status parser snapshots llama-server's
    // ``prompt processing ... progress = X`` log lines onto the
    // manager; the poller fetches them every 500ms and feeds the
    // renderer's setPrefillProgress.
    if (s.stage === 'prefill') {
      import('./prefill-progress.js').then(m => {
        m.startPrefillPolling(_primaryRenderer, app.state.currentModel || '');
      });
    }
    // Cold model loads + hot swaps spend 30-120s before the first
    // content delta. The /api/engine/v2/load_progress snapshot is
    // seeded at start() with a recent-median-derived expected_s;
    // poll every 500ms while the stage is open so the user sees a
    // soft "Loading deepseek-v3 · 14s of ~30s" bar instead of a
    // hung-looking spinner. Same pattern as prefill above.
    if (s.stage === 'model_load' || s.stage === 'model_swap') {
      import('./load-progress.js').then(m => {
        m.startLoadPolling(_primaryRenderer, app.state.currentModel || '');
      });
    }
  }
  if (meta.stage_complete) {
    // For the 1-day MVP we don't paint a "in 4.2s" tail — the next
    // stage_start (or the first content delta) will replace the
    // status text moments later, so the duration would flash by
    // unread. Log for diagnostics so future "where did time go?"
    // tooling can read the durations from devtools console.
    const c = meta.stage_complete;
    if (typeof console !== 'undefined' && console.debug) {
      console.debug(
        '[stage_complete]', c.stage,
        c.success === false ? `FAILED (${c.error || 'unknown'})` : 'ok',
        `${c.duration_ms}ms`,
        c.request_id ? `req=${c.request_id}` : '',
      );
    }
    if (c.stage === 'prefill') {
      import('./prefill-progress.js').then(m => m.stopPrefillPolling());
    }
    if (c.stage === 'model_load' || c.stage === 'model_swap') {
      import('./load-progress.js').then(m => m.stopLoadPolling());
    }
  }
  if (meta.model_thinking_delta) {
    _primaryRenderer.appendModelThinking(meta.model_thinking_delta);
  }
  const hasPerfMeta = [
    meta.tokens_per_second,
    meta.context_length,
    meta.context_used,
    meta.prompt_tokens,
    meta.eval_tokens,
    meta.ttft_ms,
    meta.total_duration_ms,
    meta.eval_duration_ms,
  ].some(v => v != null);
  if (hasPerfMeta) {
    _primaryRenderer.updateStreamMetrics({
      tokens_per_second: meta.tokens_per_second,
      context_length: meta.context_length,
      context_used: meta.context_used,
      prompt_tokens: meta.prompt_tokens,
      eval_tokens: meta.eval_tokens,
      reasoning_tokens: meta.reasoning_tokens,
      prompt_tokens_evaluated: meta.prompt_tokens_evaluated,
      prompt_tokens_cached: meta.prompt_tokens_cached,
      prompt_tokens_cache_write: meta.prompt_tokens_cache_write,
      kv_reuse: meta.kv_reuse,
      kv_void_cause: meta.kv_void_cause,
      ttft_ms: meta.ttft_ms,
      total_duration_ms: meta.total_duration_ms,
      eval_duration_ms: meta.eval_duration_ms,
      prompt_tokens_estimated: meta.prompt_tokens_estimated,
      eval_tokens_estimated: meta.eval_tokens_estimated,
    });
  }
  if (meta.knowledge_pack) {
    _pendingKnowledgePack = meta.knowledge_pack;
  }
  if (meta.flow_name) _primaryRenderer.setStreamFlowName(meta.flow_name);
  if (meta.group_speaker) {
    _primaryRenderer.setStreamSpeaker(meta.group_speaker);
    // Auto-swap the avatar viewport's active speaker to whoever is
    // actually responding this turn. No-op outside group voice calls
    // (onSpeakerSwitch early-returns if avatar isn't a group, the name
    // doesn't match a member, or that character is already active).
    // Dynamic import keeps avatar.js (Three.js, VRM) out of the chat
    // module's load graph for users who never open a voice call.
    import('../avatar.js').then(m => m.onSpeakerSwitch?.(meta.group_speaker)).catch(() => {});
  }
  if (meta.regex_transformed) _primaryRenderer.replaceStreamedContent(meta.regex_transformed);

  if (Array.isArray(meta.world_events) && meta.world_events.length) {
    // World-system events (rolls, tracker shifts, sheet renders) — draw
    // inline cards now, persist on the node so they survive reload.
    import('./world-panel.js').then(m => m.handleWorldEvents(meta.world_events)).catch(() => {});
    _pendingWorldEvents = _pendingWorldEvents.concat(meta.world_events);
  }

  if (meta.status === 'lorebook_mutations' && meta.mutations) {
    const session = getActiveSession();
    if (session) {
      if (!session.lorebook) session.lorebook = [];
      for (const m of meta.mutations) {
        if (!m.entry) continue;
        if (m.action === 'create') {
          session.lorebook.push(m.entry);
        } else if (m.action === 'update') {
          const idx = session.lorebook.findIndex(e => e.id === m.entry.id);
          if (idx >= 0) Object.assign(session.lorebook[idx], m.entry);
        } else if (m.action === 'delete') {
          session.lorebook = session.lorebook.filter(e => e.id !== m.entry.id);
        }
      }
    }
  }

  if (meta.confidence !== undefined) {
    import('../analytical.js').then(m => m.updateConfidence(meta.confidence)).catch(() => {});
  }

  // Flow bar
  if (meta.phases && meta.phases.length > 0) {
    import('../flow-bar.js').then(m => {
      if (!_flowBarStarted) {
        _flowBarStarted = true;
        m.onExecutionStart(meta.flow_name || '', meta.phases);
      } else {
        m.onPhaseUpdate(meta.phases);
      }
    }).catch(() => {});
  }

  // Agentic mode — switch the inspector to the task view AND project the
  // streamed task metadata (plan, progress, current step, autonomy) onto
  // the existing task-view DOM. The deliver step also emits per-artifact
  // cards under meta.delivery.kind === "artifact_card".
  if (meta.mode === 'agentic') {
    if (meta.delivery && meta.delivery.kind === 'artifact_card') {
      // Intermediate visuals (slide images, charts produced mid-build) are
      // inputs to a later assembly step — keep them out of the chat transcript
      // (no inline card, not saved on the node); the agentic inspector still
      // shows them.
      const intermediate = !!meta.delivery.payload?.intermediate;
      if (!intermediate) {
        const card = _agenticArtifactPayloadToCard(meta.delivery.payload);
        if (card) {
          _pushPendingAgenticArtifactCard(card);
          if (_primaryRenderer && typeof _primaryRenderer.renderStreamingToolCard === 'function') {
            _primaryRenderer.renderStreamingToolCard(card);
          }
        }
      }
    }
    import('../agentic.js').then(m => {
      m.showAgenticTaskView();
      if (m.renderAgenticTaskMeta) m.renderAgenticTaskMeta(meta);
      if (meta.delivery && meta.delivery.kind === 'artifact_card' && m.renderAgenticArtifactCard) {
        m.renderAgenticArtifactCard(meta.delivery.payload);
      }
    }).catch(() => {});
  }
}

let _flowBarStarted = false;
// Tracks whether the current stream is emitting unified tool events.
// Once tool_start fires, legacy tool_call pills stop rendering so we don't
// show both systems at once. Reset on each new stream.
let _unifiedToolActive = false;
// YouTube result_metadata captured from tool_call meta during the stream.
// Attached to the assistant node at stream-complete so the discovery cards
// survive page reloads (renderer.js reads node.youtubeData).
let _pendingYouTubeMeta = null;
// A/B test metadata from the load-balancer's ab_test strategy. Same
// pattern as _pendingYouTubeMeta — captured mid-stream, attached to
// the finalized assistant node so renderer.js (line 585) can show
// per-message 👍/👎 vote buttons. Only surfaces for users who've
// configured an ab_test load balancer.
let _pendingAbTest = null;
// Agentic delivery cards captured during the stream so artifact cards can be
// persisted onto the finalized assistant node and survive reloads.
let _pendingAgenticArtifactCards = [];
// Chain planner state for adaptive multi-tool runs. Held across
// chain / chain_step events so "Step N/total" labels can reference
// the announced total, and so per-step indicators can be rendered.
let _streamChainState = null;
// Knowledge-pack injection metadata captured from the final SSE chunk's
// augmentum.knowledge_pack field. Persisted to the assistant node so the
// "📚 Searched X — N sources" footer chip survives renders. None when
// retrieval was a true no-op (no pack bound for this session).
let _pendingKnowledgePack = null;
// World-system events (rolls / tracker shifts / sheets) captured from
// augmentum.world_events chunks; attached to the assistant node at stream
// finalize so inline cards survive reload.
let _pendingWorldEvents = [];
// Companion (becca_direct) tool dispatch — the backend's TagSieve emits
// becca_tool_call / becca_tool_result pairs with no id, in strict serial
// order (one tool runs at a time). Synthesize card ids client-side and
// pair call→result FIFO. Reset on each new stream.
let _beccaToolSeq = 0;
const _beccaOpenToolIds = [];

// Poll the image queue's progress endpoint while one of HER image
// tool calls is in flight, feeding the unified tool card's progress
// bar. Auto-stops on result, or the ~11-minute backstop (the image
// tool's own timeout is 600s).
let _beccaImagePollTimer = null;
let _beccaImagePollTicks = 0;

function _startBeccaImagePoll(id) {
  _stopBeccaImagePoll();
  _beccaImagePollTicks = 0;
  _beccaImagePollTimer = setInterval(async () => {
    if (++_beccaImagePollTicks > 550) { _stopBeccaImagePoll(); return; }
    try {
      const resp = await fetch('/api/image/generation-status', { credentials: 'same-origin' });
      if (!resp.ok) return;
      const s = await resp.json();
      if (!s.active) return;
      const ev = { id };
      if (typeof s.steps_done === 'number' && s.steps_total) {
        ev.percent = (s.steps_done / s.steps_total) * 100;
        ev.message = `${s.stage || 'Generating'} · ${s.steps_done}/${s.steps_total}`;
      } else if (s.stage) {
        ev.message = s.stage;
      } else if (s.queue_size > 0) {
        ev.message = `Queued (${s.queue_size} ahead)`;
      }
      if (ev.message || typeof ev.percent === 'number') {
        _primaryRenderer.renderToolProgress(ev);
      }
    } catch (_) { /* polling is best-effort */ }
  }, 1200);
}

function _stopBeccaImagePoll() {
  if (_beccaImagePollTimer) {
    clearInterval(_beccaImagePollTimer);
    _beccaImagePollTimer = null;
  }
}

// One-line human context for a becca tool card — the first short arg
// value ("dune audiobook") beats a JSON dump of the whole args object.
function _beccaToolContext(args) {
  if (!args || typeof args !== 'object') return '';
  for (const v of Object.values(args)) {
    if (typeof v === 'string' && v.trim()) return v.trim().slice(0, 80);
  }
  return '';
}

function _handlePrimaryStreamComplete(session, result, opts = {}) {
  app.state.isStreaming = false;
  _flowBarStarted = false;
  _unifiedToolActive = false;
  _streamChainState = null;
  _beccaOpenToolIds.length = 0;
  // Defensive: if the stream aborts before an assistant node is
  // created, the pending attach below never runs — clear these now
  // so stale ab_test / youtube meta can't bleed into the next turn.
  if (result?.aborted) {
    _pendingAbTest = null;
    _pendingYouTubeMeta = null;
    _pendingAgenticArtifactCards = [];
    _pendingKnowledgePack = null;
    _pendingWorldEvents = [];
  }

  // Complete flow bar
  import('../flow-bar.js').then(m => m.onExecutionComplete()).catch(() => {});

  if (!_primaryRenderer) return;

  if (result.aborted) {
    _primaryRenderer.finalizeStreaming();
    return;
  }

  // Get streamed content and create assistant node. Pull the optional hidden
  // avatar motion tag ([motion:xxx]) out first so it's never saved or rendered;
  // the cue is dispatched after finalize to animate her on-screen avatar.
  const _streamedRaw = _primaryRenderer.getStreamingRawContent() || '';
  const _motion = extractMotionCue(_streamedRaw);
  const rawContent = _motion.text;
  // Pull tool cards up-front so we can create the assistant node even
  // when the model produced no prose (direct-invoke image_search etc.) —
  // without this, captured tool cards have nowhere to land and vanish.
  const _pendingToolCards = _primaryRenderer.collectToolCards();
  // Narrative internal-tool activity (recall + lorebook) — persisted onto
  // the assistant node so the collapsed chip survives re-render on THIS
  // message (only). Full history goes to the inspector request-log.
  const _pendingNarrativeActivity = _primaryRenderer.collectNarrativeActivity();

  // Continuation: streamingEl IS the existing assistant node. The
  // renderer's dataset.rawContent has [partial + streamed] concatenated.
  // Just write that back to the node — no new node, no parent attach,
  // no activeLeafId move. Tool cards / reasoning / metrics from the
  // continuation turn merge onto the existing node so the round-trip
  // story stays coherent.
  if (opts.continueLastAssistant) {
    const existingNode = session.tree?.[session.activeLeafId];
    if (existingNode && existingNode.role === 'assistant') {
      existingNode.content = rawContent;
      const reasoning = _primaryRenderer.collectReasoningData();
      if (reasoning && reasoning.thinking) {
        // Continuations run with thinking disabled (Anthropic / DeepSeek
        // / llama-server all require it), so this is unusual — but if a
        // backend emits reasoning anyway, preserve it appended to any
        // prior thinking on the node.
        const prior = (existingNode.reasoning && existingNode.reasoning.thinking) || '';
        existingNode.reasoning = {
          ...(existingNode.reasoning || {}),
          thinking: prior + reasoning.thinking,
        };
      }
      if (_pendingToolCards.length) {
        existingNode.toolCards = [...(existingNode.toolCards || []), ..._pendingToolCards];
      }
      if (_pendingNarrativeActivity.length) {
        existingNode.narrativeActivity = [
          ...(existingNode.narrativeActivity || []), ..._pendingNarrativeActivity,
        ];
      }
      _primaryRenderer.finalizeStreaming(session);
      // syncNow on continuation finalize — same race protection as the
      // fresh-node branch below; a continuation can also carry costly
      // state (extra prose, tool cards) that the debounce window risks.
      sessionStore.syncNow(session.id);
      memoryGlow.memGlowIdle();
      memoryGlow.memStartPolling();
      const _s = getSettings();
      if (_s.voiceAutoRead && rawContent) {
        tts.ttsProgressiveFinish();
      }
      return;
    }
    // Fall through to fresh-node creation if the existing node was lost.
  }

  if (rawContent || _pendingAgenticArtifactCards.length || _pendingToolCards.length) {
    const parentId = session.activeLeafId;
    const assistantNode = tree.addChildNode(session, parentId, 'assistant', rawContent);

    // Group chat: tag the assistant node with the speaker so per-turn TTS
    // can resolve the speaker's own voice (otherwise all members share the
    // first member's voice).
    const speaker = _primaryRenderer.getStreamSpeaker();
    if (speaker) assistantNode.speakerName = speaker;

    // Store reasoning data — UARF phases, model-native thinking, or both.
    // Thinking-only (no phases) still persists so reasoning models round-
    // trip across refreshes and server restarts.
    const reasoning = _primaryRenderer.collectReasoningData();
    if (reasoning && ((reasoning.phases && reasoning.phases.length > 0) || reasoning.thinking)) {
      assistantNode.reasoning = reasoning;
    }

    // Store generation metrics
    const m = _primaryRenderer._streamMetrics;
    if (m.tps > 0) assistantNode.tokens_per_second = m.tps;
    if (m.promptTokens > 0) assistantNode.prompt_tokens = m.promptTokens;
    if (m.evalTokens > 0) assistantNode.eval_tokens = m.evalTokens;
    if (m.contextLen > 0) {
      assistantNode.context_length = m.contextLen;
      assistantNode.context_used = m.contextUsed;
    }
    if (m.promptTokensEvaluated > 0) {
      assistantNode.prompt_tokens_evaluated = m.promptTokensEvaluated;
    }
    if (m.promptTokensCached > 0) {
      assistantNode.prompt_tokens_cached = m.promptTokensCached;
    }
    if (m.promptTokensCacheWrite > 0) {
      assistantNode.prompt_tokens_cache_write = m.promptTokensCacheWrite;
    }
    if (m.kvReuse) assistantNode.kv_reuse = m.kvReuse;
    if (m.kvVoidCause) assistantNode.kv_void_cause = m.kvVoidCause;
    if (m.reasoningTokens > 0) {
      assistantNode.reasoning_tokens = m.reasoningTokens;
    }
    if (m.ttftMs > 0) assistantNode.ttft_ms = m.ttftMs;
    if (m.totalDurationMs > 0) assistantNode.total_duration_ms = m.totalDurationMs;
    if (m.evalDurationMs > 0) assistantNode.eval_duration_ms = m.evalDurationMs;
    // Persist the estimated flags so the ~ marker survives reloads.
    // Only stamp truthy values so older nodes keep clean (no flag = no marker).
    if (m.promptTokensEstimated) assistantNode.prompt_tokens_estimated = true;
    if (m.evalTokensEstimated) assistantNode.eval_tokens_estimated = true;

    if (_pendingYouTubeMeta) {
      assistantNode.youtubeData = _pendingYouTubeMeta;
      _pendingYouTubeMeta = null;
    }

    if (_pendingKnowledgePack) {
      assistantNode.knowledgePack = _pendingKnowledgePack;
      _pendingKnowledgePack = null;
    }
    if (_pendingWorldEvents.length) {
      assistantNode.world_events = (assistantNode.world_events || []).concat(_pendingWorldEvents);
      _pendingWorldEvents = [];
    }

    // Persist unified tool cards (image_search, browse_fetch, etc.) so
    // they survive renderMessages() rebuilds — without this they vanish
    // on the next user turn, branch nav, or page refresh.
    if (_pendingToolCards.length) {
      assistantNode.toolCards = _pendingToolCards;
    }
    if (_pendingNarrativeActivity.length) {
      assistantNode.narrativeActivity = _pendingNarrativeActivity;
    }

    if (_pendingAbTest) {
      assistantNode.ab_test = _pendingAbTest;
      _pendingAbTest = null;
    }

    if (_pendingAgenticArtifactCards.length) {
      assistantNode.agenticArtifactCards = _pendingAgenticArtifactCards.map((card) => ({
        ...card,
        preview: card.preview && typeof card.preview === 'object' ? { ...card.preview } : card.preview,
        actions: Array.isArray(card.actions)
          ? card.actions.map((action) => (
            action && typeof action === 'object'
              ? {
                ...action,
                payload: action.payload && typeof action.payload === 'object'
                  ? { ...action.payload }
                  : action.payload,
              }
              : action
          ))
          : [],
      }));
      _pendingAgenticArtifactCards = [];
    }

    session.activeLeafId = assistantNode.id;
    _primaryRenderer.setStreamingNodeId(assistantNode.id);
  }

  // Pass session so finalizeStreaming can render the branch-swipe indicator
  // when this node has siblings (e.g. after a regenerate).
  _primaryRenderer.finalizeStreaming(session);

  // Avatar motion cue — drive her on-screen avatar from the model's own hidden
  // tag (already stripped from text above). becca-presence listens and animates
  // only when she's actually shown, mapping the cue to roles so the user's
  // ratings / disables / uploaded clips govern which clip plays.
  if (_motion.cue) {
    try {
      document.dispatchEvent(new CustomEvent('augmentum:motion-cue', { detail: { cue: _motion.cue } }));
    } catch (_) { /* avatar absent — ignore */ }
  }
  // syncNow() instead of save() — eliminates the 500ms debounce window
  // during which a fast refresh after a long-running agentic chain (10+
  // minute storybook generation) used to silently lose the finalized
  // assistant node (the prose + toolCards + agenticArtifactCards). The
  // unload-time flush() bailed at 60KB and the regular debounced sync
  // hadn't fired yet, so the user came back to just their prompt.
  sessionStore.syncNow(session.id);

  // Memory glow — start polling for new memories
  memoryGlow.memGlowIdle();
  memoryGlow.memStartPolling();

  // Auto-read TTS
  const s = getSettings();
  if (s.voiceAutoRead && rawContent) {
    tts.ttsProgressiveFinish();
  }

  // Title generation
  _maybeGenerateTitle(session);
}

async function _maybeGenerateTitle(session) {
  if (!session || session.title !== 'New Chat') return;
  if (!session.rootId) return;
  const root = session.tree[session.rootId];
  if (!root || root.role !== 'user') return;

  const text = root.content;
  session.title = text.slice(0, 50) + (text.length > 50 ? '...' : '');
  sessionStore.save(session.id);
  renderSessionList();

  try {
    const resp = await fetch('/api/ui/generate-title', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, model: app.state.currentModel || '' }),
    });
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.title && data.title !== 'New Chat') {
      session.title = data.title;
      sessionStore.save(session.id);
      renderSessionList();
    }
  } catch { /* truncated title is fine */ }
}

function _initGlobalListeners() {
  // --- PRIMARY SEND PIPELINE ---
  // This is the main entry point for sending messages from the primary surface.
  // app.js dispatches augmentum:send when the user clicks Send or presses Enter.
  document.addEventListener('augmentum:send', (e) => {
    const text = e.detail?.text ?? '';
    const images = e.detail?.images || [];
    const docs = e.detail?.docs || [];
    _primarySend(text, images, docs);
  });

  // Inline stall banner — Abort & retry. Renderer dispatches this when
  // the user clicks the banner button. We abort the stuck stream, then
  // fire a Continue from the partial we already streamed so the user
  // doesn't lose what's there. Falls back to a clean state if there's
  // no partial to continue from.
  document.addEventListener('augmentum:stall-abort-retry', () => {
    if (_activeStream) _activeStream.abort();
    const session = getActiveSession();
    // Wait for onComplete({aborted:true}) to settle the renderer state
    // before re-arming a new stream. 100ms is more than enough — the
    // abort path is synchronous up to the AbortController firing.
    setTimeout(() => {
      if (!session) return;
      const lastNode = session.tree?.[session.activeLeafId];
      if (!lastNode || lastNode.role !== 'assistant') return;
      // Clear the interrupted marker added by the abort path; we're
      // about to continue, not finalize.
      if (lastNode.interrupted) {
        delete lastNode.interrupted;
        delete lastNode.error_message;
        delete lastNode.error_kind;
      }
      app.state.isStreaming = true;
      _primaryStreamResponse(session, { continueLastAssistant: true });
    }, 100);
  });

  // Stop button — aborts the in-flight primary chat stream.
  // Also opportunistically cancels backgrounded jobs (image gen,
  // app-builder build) the chat may have kicked off, since those
  // continue server-side after the chat stream is closed.
  document.addEventListener('augmentum:stop', () => {
    if (_activeStream) {
      _activeStream.abort();
    }
    // Multi-model fan-out — aborts every in-flight compare stream;
    // partials persist as interrupted sibling nodes.
    multiModel.abortFanout();
    if (_primaryRenderer) {
      _primaryRenderer.finalizeStreaming();
    }
    // Stream abort routes through onComplete({aborted:true}), not onError,
    // so the progressive TTS pipeline won't self-cancel. Kill it here.
    tts.ttsProgressiveCancel();
    tts.ttsStopCurrent();
    app.state.isStreaming = false;
    // Best-effort: tell the server to release any in-flight side jobs.
    fetch('/api/image/cancel', { method: 'POST' }).catch(() => { /* idempotent */ });
    fetch('/api/artifacts/build-cancel', { method: 'POST' }).catch(() => { /* idempotent */ });
  });

  // Voice message injection (add to chat tree without sending to backend).
  //
  // Route to the session the call was started from, not whatever session is
  // currently open. Without this, minimizing the call and switching to a
  // different chat would drop voice turns into the wrong session — and worse,
  // the server-side VoiceSession is still using the origin session's context,
  // so the assistant response would reflect Chat A's state written into Chat C.
  document.addEventListener('augmentum:voice-message', (e) => {
    const { role, text, sessionId } = e.detail || {};
    if (!text) return;
    const session = (sessionId && sessionStore.get(sessionId)) || getActiveSession();
    if (!session) return;
    const parentId = session.activeLeafId || null;
    const node = tree.addChildNode(session, parentId, role || 'user', text);
    session.activeLeafId = node.id;
    if (role === 'user') _maybeGenerateTitle(session);
    sessionStore.save(session.id);
    // Only re-render if the target session is the one currently on screen.
    // Otherwise the insertion is silent — user sees the new turns when they
    // navigate back to the origin session.
    const activeId = sessionStore.getActiveId();
    if (session.id === activeId) renderMessages();
  });

  // New session trigger
  document.addEventListener('augmentum:new-session', () => {
    const id = sessionStore.create(app.state.mode || 'passthrough');
    app.state.currentSessionId = id;
    sessionStore.setActiveId(id);  // dispatches augmentum:session-changed
    renderSessionList();
    renderMessages();
  });

  // Mode change — switch to a session matching the new mode, re-render list.
  //
  // Coder stays OUT of this set: its "sessions" are workspace containers,
  // not sessionStore rows, so forMode("coder") is empty and the fallback
  // branch would setActiveId(null) and clobber whatever chat/narrative
  // session was live. Narrative IS handled — it uses sessionStore like
  // the chat modes do, and restoring the last-active narrative session
  // on mode entry is the whole point (without it, clicking the narrative
  // orb shows the generic "What would you like to do?" empty state over
  // the stale passthrough session, stranding the user until they pick a
  // character from the left panel).
  const _CHAT_HANDLED_MODES = new Set(['passthrough', 'analytical', 'agentic', 'narrative']);

  document.addEventListener('augmentum:mode-changed', (e) => {
    const newMode = e.detail?.mode;
    if (!newMode) return;

    // If the new mode isn't one we own, do NOTHING. Don't touch the
    // active session, don't renderMessages, don't renderSessionList.
    // The surface that actually owns this mode (narrative / coder) will
    // manage its own view, and the existing active session (which we
    // weren't asked about) is left intact so the user's place is
    // preserved when they swap back.
    if (!_CHAT_HANDLED_MODES.has(newMode)) return;

    // Update renderer mode
    if (_primaryRenderer) _primaryRenderer.setMode(newMode);

    // Crossing mode boundaries shows a different session population, so
    // the "Show older" progression from the previous mode's list no longer
    // reflects user intent. Reset the render limit so the new list starts
    // fresh at its default batch size.
    _sessionRenderLimit = _SESSION_RENDER_BATCH;

    // Find the right session for this mode and switch to it.
    const activeId = sessionStore.getActiveId();
    const currentSession = activeId ? sessionStore.get(activeId) : null;
    if (currentSession && (currentSession.mode || 'passthrough') === newMode) {
      // Already on a matching session — just re-render the list
      renderSessionList();
      return;
    }

    // Restoration order:
    //   1. The session the user was LAST VIEWING in this mode — recorded
    //      automatically by sessionStore.setActiveId. This is the fix for
    //      "swap to chat, swap back, lands on newest narrative instead of
    //      the one I was reading". Feels like Cmd-Tab between apps.
    //   2. The most-recently-used session in this mode — fallback when
    //      the per-mode memory is empty (first time in mode this session,
    //      or the remembered session was deleted while away).
    //   3. Empty state — no sessions exist for this mode at all.
    const lastForMode = sessionStore.getLastActiveForMode(newMode);
    if (lastForMode) {
      switchSession(lastForMode);
      return;
    }
    const modeSessions = sessionStore.forMode(newMode);
    if (modeSessions.length > 0) {
      switchSession(modeSessions[0].id);
    } else {
      // No sessions for this mode — clear display, show empty state
      app.state.currentSessionId = null;
      sessionStore.setActiveId(null);  // dispatches augmentum:session-changed
      renderMessages();
      renderSessionList();
    }
  });

  // Session-list refresh on any active-session change. Fires for every caller
  // of sessionStore.setActiveId (including non-primary ChatSurface.activate()
  // when the user switches tabs), so the left panel always highlights the
  // correct session without every callsite having to remember to call
  // renderSessionList itself. Narrative mode has its own listener in
  // narrative/index.js that refreshes the character grid + recent-chats
  // strip; this one owns the plain session list.
  document.addEventListener('augmentum:session-changed', (e) => {
    const currentMode = app.state.mode || 'passthrough';
    // World drawer: mounts only when the session's card declares a
    // world manifest (server says active) — narrative sessions only.
    if (currentMode === 'narrative') {
      import('./world-panel.js')
        .then(m => m.ensureWorldPanel(e.detail?.sessionId || ''))
        .catch(() => {});
    }
    // Offer chips are scoped to the chat that produced them, and switching
    // sessions wipes them out of the DOM with the message list — re-backfill
    // so the new session's chips (and only those) come back. Deferred a tick
    // because renderMessages() below clears the container; re-inserting before
    // that would just have them wiped again. Ahead of the early return so it
    // still runs for narrative/coder, whose sessions raise offers too.
    setTimeout(() => {
      try { _primaryRenderer?._offerFeedHandle?.rescope(); } catch { /* best-effort */ }
    }, 0);
    if (currentMode === 'narrative' || currentMode === 'coder') return;
    app.state.currentSessionId = e.detail?.sessionId || null;
    renderSessionList();
    renderMessages();
  });

  // Cross-device reconcile re-rendered the session metadata (new/renamed/
  // deleted sessions from another device). Refresh the passthrough sidebar;
  // narrative/coder keep their own lists and refresh on next navigation.
  document.addEventListener('augmentum:sessions-reconciled', (e) => {
    const currentMode = app.state.mode || 'passthrough';
    if (currentMode === 'narrative' || currentMode === 'coder') return;
    renderSessionList();
    // A stale-sync union-merge may have grafted another device's turns
    // into the session currently on screen — repaint the message pane so
    // they appear now, not on the next navigation.
    const mergedIds = e.detail?.mergedIds;
    if (Array.isArray(mergedIds) && mergedIds.includes(sessionStore.getActiveId())) {
      renderMessages();
    }
  });

  // Page unload — sync active session
  window.addEventListener('beforeunload', () => {
    sessionStore.flush();
  });
}

// ---------------------------------------------------------------------------
// Legacy API functions (used by the `chat` export object)
// ---------------------------------------------------------------------------

function createSession(mode) {
  const modeStr = (typeof mode === 'string' && mode) ? mode : (app.state.mode || 'passthrough');
  const id = sessionStore.create(modeStr);
  renderSessionList();
  switchSession(id);  // updates state.currentSessionId + dispatches event
  return id;
}

function deleteSession(id) {
  const session = sessionStore.get(id);
  const name = session?.title || 'this chat';
  const msgCount = session?.tree ? Object.keys(session.tree).length : 0;
  const detail = msgCount > 0 ? ` (${msgCount} messages)` : '';
  if (!confirm(`Delete "${name}"${detail}? This cannot be undone.`)) return;

  const deletedMode = sessionStore.delete(id);
  if (sessionStore.getActiveId() === id || !sessionStore.get(sessionStore.getActiveId())) {
    const currentMode = app.state.mode || 'passthrough';
    const modeMatch = sessionStore.forMode(currentMode);
    if (modeMatch.length > 0) {
      switchSession(modeMatch[0].id);
      return;
    }
    app.state.currentSessionId = null;
    sessionStore.setActiveId(null);  // dispatches augmentum:session-changed
    renderMessages();
  }
  renderSessionList();
}

function switchSession(id) {
  if (!sessionStore.get(id)) return;

  // Abort current stream
  if (_activeStream) {
    _activeStream.abort();
    _activeStream = null;
  }
  app.state.isStreaming = false;

  // Update the state mirror BEFORE setActiveId — the latter dispatches
  // augmentum:session-changed synchronously and listeners read state.currentSessionId.
  app.state.currentSessionId = id;
  sessionStore.setActiveId(id);  // dispatches augmentum:session-changed
  renderSessionList();

  // Lazy-load full session data if needed
  sessionStore.ensureLoaded(id).then(() => renderMessages());

  // Kick the engine's KV resume ladder for this session so its prefix
  // is being restored/replayed while the user reads and types. Server
  // dedupes rapid flips; a cold/not-ready outcome is a normal 200.
  _fireKvResume(id);

  // Focus input on desktop
  if (window.innerWidth >= 768 && app.dom.chatInput) {
    app.dom.chatInput.focus();
  }
}

function getActiveSession() {
  const id = sessionStore.getActiveId();
  return id ? sessionStore.get(id) : null;
}

// Session id we last asked the resume ladder about, with a short
// client-side guard so flicking through the session list doesn't spam
// the endpoint (the server also dedupes in-flight resumes per key).
let _lastKvResumeId = '';
let _lastKvResumeTs = 0;

function _fireKvResume(sessionId) {
  const now = Date.now();
  if (sessionId === _lastKvResumeId && (now - _lastKvResumeTs) < 5000) return;
  _lastKvResumeId = sessionId;
  _lastKvResumeTs = now;
  fetch('/api/engine/v2/kv/resume', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  }).catch(() => {}); // best-effort — a cold open is always correct
}

function renderMessages() {
  if (!_primaryRenderer) return;
  const session = getActiveSession();
  _primaryRenderer.renderMessages(session);

  // Restore reasoning inspector
  if (session) {
    const path = tree.getActivePath(session);
    const lastReasoning = [...path].reverse().find(n => n.role === 'assistant' && n.reasoning);
    if (lastReasoning?.reasoning) {
      restoreReasoningFromStored(lastReasoning.reasoning);
    } else {
      resetReasoningState();
    }
  }
}

/**
 * How many sessions we render up-front before showing a "Show older…"
 * footer. Power users can accumulate thousands of chats — rendering them
 * all on every setActiveId (our session-list refresh listener fires on
 * every surface tab switch, among other things) causes visible jank and
 * wastes memory on off-screen list items. The footer lazily reveals more
 * in batches so most users never pay the cost.
 *
 * Module state — _sessionRenderLimit grows when the user asks for more.
 * It resets to the default on mode change (below) because switching modes
 * typically means switching to a different list entirely.
 */
const _SESSION_RENDER_BATCH = 200;
let _sessionRenderLimit = _SESSION_RENDER_BATCH;

function renderSessionList() {
  // Delegate to the existing session list renderer in the left panel.
  // This will eventually move to a shared component.
  const list = app.dom.sessionsView?.querySelector('#session-list');
  if (!list) return;
  list.innerHTML = '';

  const currentMode = app.state.mode || 'passthrough';
  const sorted = sessionStore.forMode(currentMode);

  if (sorted.length === 0) {
    const { modeLabel } = app;
    list.innerHTML = `<div class="session-empty-hint">No chats yet. Click + to start one.</div>`;
    return;
  }

  const activeId = sessionStore.getActiveId();
  let lastGroup = '';

  const total = sorted.length;
  // Always include the active session even if it falls past the limit
  // (otherwise switching to an older chat would make the active row
  // disappear from the list). Find its index; if it's beyond the limit,
  // bump the limit to include it for this render pass only.
  const activeIdx = activeId ? sorted.findIndex(s => s.id === activeId) : -1;
  const effectiveLimit = (activeIdx >= 0 && activeIdx >= _sessionRenderLimit)
    ? activeIdx + 1
    : _sessionRenderLimit;
  const visible = sorted.slice(0, effectiveLimit);

  visible.forEach(s => {
    const group = _dateGroup(s.createdAt);
    if (group !== lastGroup) {
      lastGroup = group;
      const divider = document.createElement('div');
      divider.className = 'session-group-label';
      divider.textContent = group;
      list.appendChild(divider);
    }

    const timeLabel = _relativeTime(s.createdAt);
    const item = document.createElement('div');
    item.className = 'list-item' + (s.id === activeId ? ' active' : '');
    item.title = s.title || 'Untitled';
    item.innerHTML = `
      <svg class="list-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
      <div class="list-item-body">
        <span class="list-item-text">${escapeHtml(s.title)}</span>
        <span class="list-item-time">${escapeHtml(timeLabel)}</span>
      </div>
      <button class="message-action-btn" data-export="${s.id}" title="Export">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
      </button>
      <button class="message-action-btn" data-delete="${s.id}" title="Delete">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    `;
    item.addEventListener('click', (e) => {
      if (e.target.closest('[data-delete]') || e.target.closest('[data-export]')) return;
      switchSession(s.id);
    });
    item.querySelector('[data-export]').addEventListener('click', (e) => {
      e.stopPropagation();
      exportChat(s.id);
    });
    item.querySelector('[data-delete]').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteSession(s.id);
    });
    list.appendChild(item);
  });

  // "Show older…" footer when more sessions exist than we rendered.
  // Clicking grows the module-level limit and re-renders. The button is
  // cheap to produce and self-removes on the next render when all
  // sessions become visible.
  if (total > effectiveLimit) {
    const hidden = total - effectiveLimit;
    const showMoreBtn = document.createElement('button');
    showMoreBtn.className = 'session-show-older';
    showMoreBtn.type = 'button';
    showMoreBtn.textContent = `Show ${Math.min(hidden, _SESSION_RENDER_BATCH)} older chat${hidden === 1 ? '' : 's'}`;
    showMoreBtn.title = `${hidden} older chat${hidden === 1 ? '' : 's'} hidden — click to reveal the next batch`;
    showMoreBtn.addEventListener('click', () => {
      _sessionRenderLimit += _SESSION_RENDER_BATCH;
      renderSessionList();
    });
    list.appendChild(showMoreBtn);
  }

  document.dispatchEvent(new CustomEvent('augmentum:sessions-rendered'));
}

function stopStreaming() {
  if (_activeStream) {
    _activeStream.abort();
    _activeStream = null;
  }
  app.state.isStreaming = false;
}

function exportChat(id) {
  const data = sessionStore.export(id);
  if (!data) return;
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `chat-${id}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function exportAllChats() {
  const data = sessionStore.exportAll();
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'augmentum-chats.json';
  a.click();
  URL.revokeObjectURL(url);
}

function importChats() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.json';
  input.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      const sessions = data.sessions || data;
      await sessionStore.import(sessions);
      renderSessionList();
      // Success toast removed — the chat list visibly updates, so the
      // toast was duplicative noise. Errors still surface (load below).
    } catch (err) {
      showToast(`Couldn't import: ${err.message}`, 'error');
    }
  });
  input.click();
}

function saveSessions(touchedSessionId = null) {
  // Pass the id of the mutated session — a bare save() only writes
  // localStorage stubs; it does NOT mark anything dirty for server sync.
  sessionStore.save(touchedSessionId);
}

function updateCharacterVoice(sessionId, voice) {
  tts.updateCharacterVoice(sessionId, voice, sessionStore);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _relativeTime(ts) {
  if (!ts) return '';
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days}d ago`;
  return new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function _dateGroup(ts) {
  if (!ts) return 'Older';
  const now = new Date();
  const d = new Date(ts);
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today); yesterday.setDate(today.getDate() - 1);
  const weekAgo = new Date(today); weekAgo.setDate(today.getDate() - 7);
  if (d >= today) return 'Today';
  if (d >= yesterday) return 'Yesterday';
  if (d >= weekAgo) return 'Previous 7 days';
  return 'Older';
}

function _handleLegacySend(text, images, docs) {
  document.dispatchEvent(new CustomEvent('augmentum:send', {
    detail: { text, images, docs },
  }));
}

function _initSessionSearch() {
  const searchInput = document.getElementById('session-search');
  if (!searchInput) return;
  const apply = () => {
    const list = document.getElementById('session-list');
    if (!list) return;
    const query = searchInput.value.toLowerCase().trim();
    const children = Array.from(list.children);
    let lastVisibleDivider = null;
    let dividerHasMatch = false;
    children.forEach(node => {
      if (node.classList.contains('session-group-label')) {
        if (lastVisibleDivider && !dividerHasMatch) lastVisibleDivider.style.display = 'none';
        lastVisibleDivider = node;
        dividerHasMatch = false;
        node.style.display = '';
        return;
      }
      if (!node.classList.contains('list-item')) return;
      const text = (node.querySelector('.list-item-text')?.textContent || '').toLowerCase();
      const match = !query || text.includes(query);
      node.style.display = match ? '' : 'none';
      if (match) dividerHasMatch = true;
    });
    if (lastVisibleDivider && !dividerHasMatch) lastVisibleDivider.style.display = 'none';
  };
  searchInput.addEventListener('input', apply);
  document.addEventListener('augmentum:sessions-rendered', apply);
}

function _initSessionFooter() {
  const sessionsView = app.dom.sessionsView;
  if (sessionsView && !sessionsView.querySelector('.panel-footer')) {
    const footer = document.createElement('div');
    footer.className = 'panel-footer';
    footer.style.cssText = 'display:flex;gap:var(--space-xs)';
    footer.innerHTML = `
      <button class="btn btn-sm btn-ghost" id="export-all-chats-btn" style="flex:1" title="Export all chats">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        Export All
      </button>
      <button class="btn btn-sm btn-ghost" id="import-chats-btn" style="flex:1" title="Import chats from file">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        Import
      </button>
    `;
    sessionsView.appendChild(footer);

    footer.querySelector('#export-all-chats-btn').addEventListener('click', exportAllChats);
    footer.querySelector('#import-chats-btn').addEventListener('click', importChats);
  }
}

// ---------------------------------------------------------------------------
// Tool deliverable dispatcher
// ---------------------------------------------------------------------------
// Tools that want rich UI (panels, cards, galleries, players) return the
// relevant keys in ToolResult.metadata. The backend forwards that whole
// dict as `result_metadata` on tool_call / tool_complete events. This
// dispatcher inspects the tool name + metadata and invokes the right
// renderer, so the UI doesn't depend on the model echoing the payload
// in prose. To add a new tool, just register a handler here — no
// handler.py changes, no tool-by-tool whitelisting.

/**
 * @typedef {Object} DeliverableContext
 * @property {HTMLElement|null} messageEl  The streaming assistant message.
 * @property {HTMLElement|null} toolCard   The tool card DOM (may be null).
 */

/** @type {Object<string, (meta: object, ctx: DeliverableContext) => void>} */
const _TOOL_DELIVERABLES = {
  youtube: (meta, { messageEl }) => {
    if (!meta || !messageEl) return;
    if (meta.youtube_mode === 'search' && Array.isArray(meta.results) && meta.results.length) {
      renderYouTubeCards(meta, messageEl);
    } else if (meta.youtube_mode === 'direct' && meta.video_id) {
      import('../youtube-panel.js').then(yt => yt.openDirect(meta)).catch(() => {});
    }
  },
  // image_generation: no-op. The passthrough/analytical handlers append
  // `![Generated Image](/api/image/xxx)` to the assistant's content, so
  // the image renders inline via markdown — persisted as part of
  // node.content, with lightbox support already wired through .md-image
  // (renderer.js delegated click). Calling showGeneratedImage here used
  // to insert a duplicate .tool-image-card after the message that
  // vanished on refresh.
  image_generation: () => {},
  // image_search gallery is still rendered inside the tool card by
  // renderer.renderToolComplete — registered here as a no-op so callers
  // see the tool is known (avoids falling back to "unknown tool" logs
  // once we add logging).
  image_search: () => {},
  // build_application is long-running: its "deliverable" is the initial
  // project_progress event (rendered by project.js's build monitor),
  // not a tool_complete payload. Registered as a no-op so the
  // dispatcher doesn't warn if a final event ever shows up for it.
  build_application: () => {},
};

/**
 * Dispatch a tool's result to its deliverable renderer, if one is
 * registered. Safe to call for any tool; unknown tools are ignored.
 */
export function renderToolDeliverable(toolName, resultMetadata, ctx) {
  if (!toolName || !resultMetadata) return;
  const handler = _TOOL_DELIVERABLES[toolName];
  if (!handler) return;
  try {
    handler(resultMetadata, ctx || { messageEl: null, toolCard: null });
  } catch (err) {
    console.warn('[chat] deliverable failed', toolName, err);
  }
}

// ---------------------------------------------------------------------------
// YouTube cards, image display, lightbox, related files, save-to-library
// ---------------------------------------------------------------------------

/** Render YouTube discovery cards inline or auto-open the video panel. */
export function renderYouTubeCards(meta, messageEl) {
  if (!meta || !messageEl) return;
  const contentEl = messageEl.querySelector('.message-content');
  if (!contentEl) return;

  if (meta.youtube_mode === 'search' && meta.results) {
    const container = document.createElement('div');
    container.className = 'yt-discovery';
    for (const r of meta.results) {
      const thumbUrl = '/api/browse/image?url=' + encodeURIComponent(r.thumbnail);
      const card = document.createElement('div');
      card.className = 'yt-card cast-btn-host';
      card.dataset.videoId = r.video_id;
      card.dataset.title = r.title || '';
      card.dataset.channel = r.channel || '';
      card.innerHTML =
        '<div class="yt-card-thumb">' +
          '<img src="' + escapeHtml(thumbUrl) + '" alt="' + escapeHtml(r.title) + '" loading="lazy" onerror="this.style.display=\'none\'">' +
          (r.duration ? '<span class="yt-card-duration">' + escapeHtml(String(r.duration)) + '</span>' : '') +
        '</div>' +
        '<div class="yt-card-info">' +
          '<div class="yt-card-title">' + escapeHtml(r.title) + '</div>' +
          '<div class="yt-card-channel">' + escapeHtml(r.channel) + '</div>' +
          '<div class="yt-card-meta">' + escapeHtml(r.views || '') + (r.views && r.published ? ' \u00b7 ' : '') + escapeHtml(r.published || '') + '</div>' +
        '</div>';
      // Mount the Cast-to-TV overlay on the thumb. Helper handles its
      // own click + stopPropagation so the card's outer click handler
      // (open in panel) doesn't also fire.
      import('../cast-button.js').then(({ mountCastButton }) => {
        const watchUrl = `https://www.youtube.com/watch?v=${encodeURIComponent(r.video_id)}`;
        const castBtn = mountCastButton({
          capability: 'media.video_play@1',
          size: 'sm',
          className: 'cast-btn-on-image cast-btn-hover-reveal yt-card-cast',
          title: 'Cast to TV',
          getContent: () => ({
            contentUrl: watchUrl,
            title: r.title || 'YouTube video',
            posterUrl: r.thumbnail || '',
            contentKey: `yt:${r.video_id}`,
            metadata: {
              platform: 'youtube',
              channel: r.channel || '',
              source: 'chat-yt-search',
            },
          }),
        });
        const thumbEl = card.querySelector('.yt-card-thumb');
        if (thumbEl) thumbEl.appendChild(castBtn);
      }).catch((err) => console.warn('[chat] cast-button mount failed', err));
      container.appendChild(card);
    }
    container.addEventListener('click', (e) => {
      const card = e.target.closest('.yt-card');
      if (!card) return;
      // Cast button has its own click handler that stops propagation;
      // any other click on the card opens the YouTube panel.
      if (e.target.closest('.cast-btn')) return;
      import('../youtube-panel.js').then(yt => {
        yt.openFromSearch(card.dataset.videoId, card.dataset.title, card.dataset.channel);
      });
    });
    contentEl.appendChild(container);
  }

  if (meta.youtube_mode === 'direct' && meta.video_id) {
    import('../youtube-panel.js').then(yt => yt.openDirect(meta));
  }
}

/** Show a generated image card below the message. */
export function showGeneratedImage(imageUrl, container) {
  const msgEl = container.closest('.message');
  if (!msgEl) return;
  if (msgEl.parentElement?.querySelector(`.tool-image-card[data-src="${CSS.escape(imageUrl)}"]`)) return;
  const card = document.createElement('div');
  card.className = 'tool-image-card';
  card.dataset.src = imageUrl;
  card.innerHTML = `<img src="${escapeHtml(imageUrl)}" alt="Generated image" loading="lazy" onerror="this.parentElement.style.display='none'">`;
  card.addEventListener('click', () => openImageLightbox(imageUrl));
  msgEl.insertAdjacentElement('afterend', card);
}

/** Fetch the most recent image from the library. */
export async function fetchAndShowLatestImage(container) {
  try {
    const resp = await fetch('/api/image/history?limit=1&sort=newest');
    if (!resp.ok) return;
    const data = await resp.json();
    const entry = (data.entries || [])[0];
    if (!entry) return;
    showGeneratedImage(`/api/image/${entry.image_id}`, container);
  } catch { /* silent */ }
}

/** Simple lightbox overlay for viewing generated images full-size. */
export function openImageLightbox(src) {
  let overlay = document.getElementById('image-lightbox');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'image-lightbox';
    overlay.innerHTML = '<img src="" alt="Full size image">';
    overlay.addEventListener('click', () => overlay.classList.remove('active'));
    document.body.appendChild(overlay);
  }
  overlay.querySelector('img').src = src;
  overlay.classList.add('active');
}

/**
 * Query the file index for files related to the user's message text and render
 * them as clickable chips below the chat input. Fire-and-forget.
 */
export async function surfaceRelatedFiles(text) {
  if (!text || text.length < 20) return;
  const strip = document.getElementById('related-files-strip');
  if (!strip) return;
  try {
    const q = encodeURIComponent(text.slice(0, 200));
    const resp = await fetch(`/api/files/search?q=${q}&limit=3`);
    if (!resp.ok) { strip.classList.add('hidden'); return; }
    const { files } = await resp.json();
    if (!files || !files.length) { strip.classList.add('hidden'); return; }
    strip.innerHTML = files.map(f => {
      const icon = _fileIcon(f.source);
      const label = f.name.length > 20 ? f.name.slice(0, 18) + '\u2026' : f.name;
      return `<button class="file-chip" data-file-id="${escapeHtml(f.id)}" data-name="${escapeHtml(f.name)}" title="${escapeHtml(f.description || f.name)}">` +
        `<span class="file-chip-icon">${icon}</span>` +
        `<span>${escapeHtml(label)}</span>` +
        `</button>`;
    }).join('');
    strip.classList.remove('hidden');
  } catch {
    strip.classList.add('hidden');
  }
}

function _fileIcon(source) {
  const map = { artifacts: '\ud83d\udcc4', images: '\ud83d\uddbc', documents: '\ud83d\udccb', knowledge: '\ud83d\udcda', voices: '\ud83c\udf99', chat_images: '\ud83d\udcf7' };
  return map[source] || '\ud83d\udcc1';
}

/** Save a code block to the library as an artifact. */
export async function saveCodeBlockToLibrary(codeHeader) {
  const rawCode = decodeURIComponent(codeHeader.dataset.rawCode || '');
  if (!rawCode.trim()) { showToast('Nothing in this code block to save.', 'warning'); return; }
  const lang = (codeHeader.dataset.lang || '').toLowerCase();

  let path = 'index.html';
  let role = 'entry';
  if (lang === 'css' || lang === 'scss') { path = 'styles.css'; role = 'style'; }
  else if (lang === 'javascript' || lang === 'js') { path = 'app.js'; role = 'script'; }
  else if (lang === 'json') { path = 'data.json'; role = 'data'; }
  else if (lang === 'svg') { path = 'image.svg'; role = 'entry'; }

  let title = 'Untitled';
  const titleMatch = rawCode.match(/<title[^>]*>(.*?)<\/title>/i);
  if (titleMatch) title = titleMatch[1].trim() || title;
  else {
    const h1Match = rawCode.match(/<h1[^>]*>(.*?)<\/h1>/i);
    if (h1Match) title = h1Match[1].replace(/<[^>]+>/g, '').trim() || title;
  }

  try {
    const source = JSON.stringify({ type: 'application', name: title, files: [{ path, role, content: rawCode }] });
    const resp = await fetch('/api/artifacts/save-html', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content: rawCode,
        filename: path,
        display_name: title,
        format: lang === 'html' || lang === 'htm' ? 'html' : lang,
        source_json: source,
        session_id: sessionStore.getActiveId() || '',
      }),
    });
    if (resp.ok) {
      // Success toast removed — the library entry appears visibly on
      // its next render. Failure cases still surface below.
    } else {
      showToast("Couldn't save to your library.", 'error');
    }
  } catch {
    showToast("Couldn't save to your library.", 'error');
  }
}

// ---------------------------------------------------------------------------
// Public API — backward-compatible with the old `chat` export
// ---------------------------------------------------------------------------

export const chat = {
  createSession,
  deleteSession,
  switchSession,
  getActiveSession,
  renderMessages,
  renderSessionList,
  stopStreaming,
  exportChat,
  exportAllChats,
  importChats,
  saveSessions,
  ttsStopCurrent: tts.ttsStopCurrent,
  ttsProgressiveCancel: tts.ttsProgressiveCancel,
  ttsChatWarmup: tts.ttsChatWarmup,
  buildMessagesForAPI: tree.buildMessagesForAPI,
  updateCharacterVoice,
  getSessions: () => sessionStore.all(),
  getActiveSessionId: () => sessionStore.getActiveId(),
  /** Push the active character's narrative-panel preference into the
   *  primary renderer. Called by the narrative module's session-changed
   *  listener so swapping cards/sessions updates the default-collapsed
   *  state without having to re-render existing messages. */
  setNarrativePanelsCollapsed: (collapsed) => {
    if (_primaryRenderer && typeof _primaryRenderer.setNarrativePanelsCollapsed === 'function') {
      _primaryRenderer.setNarrativePanelsCollapsed(collapsed);
    }
  },
};
