/**
 * ChatSurface — independent chat surface with its own DOM.
 *
 * Each instance creates its own message renderer, input area, and streaming
 * pipeline. Multiple ChatSurfaces can coexist without fighting over DOM.
 *
 * Migration: The primary (first) surface still uses the legacy chat module
 * for features not yet extracted (TTS, code editing, project builder).
 * Additional surfaces use the new instance-scoped classes directly.
 */

import { Surface } from '../surface.js';
import { SurfaceRegistry } from '../surface-registry.js';
import { app, escapeHtml, showToast } from '../app.js';
import { getSettings } from '../settings.js';

import { MessageRenderer } from '../chat/renderer.js';
import { ChatInput } from '../chat/input.js';
import { ChatStream } from '../chat/stream.js';
import { sessionStore } from '../chat/sessions.js';
import * as tree from '../chat/tree.js';
import { runImpersonate, fetchActivePersona } from '../chat/impersonate.js';
import { ttsPlayMessage } from '../chat/tts.js';
import {
  toggleHtmlPreview, runCodeBlock, toggleSvgPreview, toggleCodeEdit,
  downloadCodeBlock,
} from '../chat/code-actions.js';
import {
  showAskAiPrompt, showQuickActionsMenu, autoFixCodeBlock,
} from '../chat/code-edit.js';
import { saveCodeBlockToLibrary, renderToolDeliverable } from '../chat/index.js';

function getPrimaryInputArea() {
  return document.getElementById('chat-input')?.closest('.input-area') || null;
}

export class ChatSurface extends Surface {
  static type = 'chat';

  constructor(id, config = {}) {
    super(id, config);
    this._mode = config.analytical ? 'analytical' : (config.mode || 'passthrough');
    this._sessionId = config.sessionId || null;
    this._isPrimary = config.primary || false;

    // Instance-scoped components — created on mount
    this.renderer = null;
    this.input = null;
    this.stream = null;
    // Set for the duration of a continue-in-place stream so the complete
    // handler writes back onto the existing assistant node instead of
    // branching a new one. Cleared at the top of _handleStreamComplete.
    this._continuingNodeId = null;

    // Listeners attached on mount, removed on unmount (primary only).
    // Primary adopts the legacy singleton DOM, so its effective session and
    // mode drift with the app — we mirror those here so re-focusing the
    // primary tab later restores the right state.
    this._primarySessionListener = null;
    this._primaryModeListener = null;
    // Independent (non-primary) listener — picks up left-panel session
    // clicks so a focused non-primary chat tab actually switches which
    // chat the user is reading instead of staying frozen on the session
    // it was opened with. Symmetric to NarrativeSurface's listener.
    this._sessionChangedListener = null;
  }

  mount(container) {
    super.mount(container);
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.height = '100%';
    container.style.overflow = 'hidden';

    if (this._isPrimary) {
      // Primary surface: adopt existing DOM elements for backward compat.
      // This preserves all existing event handlers, TTS, code editing, etc.
      this._mountPrimary(container);
    } else {
      // Additional surfaces: create independent DOM.
      this._mountIndependent(container);
    }
  }

  /**
   * Primary surface — adopts existing singleton DOM.
   * Used during the migration period. Once all features are extracted
   * into the chat module, this path can be removed.
   */
  _mountPrimary(container) {
    const chatScroll = document.getElementById('chat-scroll');
    const inputArea = getPrimaryInputArea();

    if (chatScroll && inputArea) {
      chatScroll.classList.remove('hidden');
      inputArea.classList.remove('hidden');
      container.appendChild(chatScroll);
      container.appendChild(inputArea);
    }

    // Mirror the global active session while primary is visible so
    // switching to another tab and back restores the user's place.
    this._sessionId = sessionStore.getActiveId();
    this._primarySessionListener = (e) => {
      if (this.isActive) this._sessionId = e.detail?.sessionId || null;
    };
    document.addEventListener('augmentum:session-changed', this._primarySessionListener);

    // Primary's legacy DOM swaps between chat-like modes via app.js setMode;
    // keep _mode in sync so _pickSurfaceForOrb and getContext report reality.
    // Gate on isActive: a non-primary chat tab focus drives a global
    // mode-changed via the surface:focus-changed → applyMode path. Without
    // this guard the primary's _mode would silently corrupt to whichever
    // chat-mode tab the user most recently visited, locking it in (audit §4.1).
    this._primaryModeListener = (e) => {
      if (!this.isActive) return;
      const m = e.detail?.mode;
      if (m && ['passthrough', 'analytical', 'agentic'].includes(m)) {
        this._mode = m;
      }
    };
    document.addEventListener('augmentum:mode-changed', this._primaryModeListener);
  }

  /**
   * Independent surface — creates its own DOM using the new classes.
   * Each instance has its own message list, input area, and streaming.
   */
  _mountIndependent(container) {
    // Create session if needed
    if (!this._sessionId) {
      this._sessionId = sessionStore.create(this._mode);
    }

    // --- Renderer ---
    this.renderer = new MessageRenderer({
      mode: this._mode,
      onAction: (action, nodeId, data) => this._handleAction(action, nodeId, data),
      highlightHooks: {},
    });
    this.renderer.createDOM(container);

    // --- Input ---
    // Pass `surface: this` so Steps 2-6 of the surface-owned composer
    // migration can route toolbar state reads through this surface instead
    // of process globals. The param is currently stored but unused.
    this.input = new ChatInput({
      surface: this,
      onSend: (text, images, docs) => this._handleSend(text, images, docs),
      onStop: () => this._handleStop(),
    });
    this.input.createDOM(container);

    // --- Stream ---
    this.stream = new ChatStream({
      surface: this,
      onContent: (text) => {
        this.renderer.appendToStreaming(text);
      },
      onMeta: (meta) => this._handleMeta(meta),
      onComplete: (result) => this._handleStreamComplete(result),
      onError: (err) => {
        showToast('Stream error: ' + (err.message || 'Unknown'), 'error');
        this.input.setStreaming(false);
      },
    });

    // Render existing messages
    const session = sessionStore.get(this._sessionId);
    if (session) {
      sessionStore.ensureLoaded(this._sessionId).then(() => {
        const loaded = sessionStore.get(this._sessionId);
        if (loaded) this.renderer.renderMessages(loaded);
      });
    }

    // Picks up left-panel session list clicks so a focused non-primary
    // chat tab re-renders for the new session instead of staying frozen
    // on its opening session. Mode gate confines updates to chat-mode
    // sessions (passthrough/analytical/agentic) so clicking a narrative
    // session while a chat tab is focused doesn't pull a story into a
    // chat tab.
    this._sessionChangedListener = (e) => {
      if (!this.isActive) return;
      const newId = e.detail?.sessionId;
      if (!newId || newId === this._sessionId) return;
      const s = sessionStore.get(newId);
      if (!s) return;
      const newMode = s.mode || 'passthrough';
      if (!['passthrough', 'analytical', 'agentic'].includes(newMode)) return;
      this._sessionId = newId;
      this._mode = newMode;
      // Re-gate the per-surface toolbar for the new mode (tools/auto-search
      // visibility differs across passthrough/analytical/agentic).
      this.input?.updateToolbarMode?.(newMode);
      this.emit('surface:titleChanged', { title: this.getTitle() });
      sessionStore.ensureLoaded(newId).then(() => {
        const loaded = sessionStore.get(newId);
        if (loaded && this.renderer) this.renderer.renderMessages(loaded);
      });
    };
    document.addEventListener('augmentum:session-changed', this._sessionChangedListener);
  }

  activate() {
    super.activate();
    // Promote this tab's session to global-active so every module subscribed
    // to augmentum:session-changed refreshes to reflect this tab (left panel
    // session list, inspector, character grid, renderers). sessionStore
    // no-ops if the id is unchanged, so re-focusing the same tab is cheap.
    if (this._sessionId) {
      sessionStore.setActiveId(this._sessionId);
    }
    // Independent surfaces own their renderer — repaint from the session in
    // case the stored tree was mutated while the tab was hidden.
    if (!this._isPrimary && this.renderer && this._sessionId) {
      const session = sessionStore.get(this._sessionId);
      if (session) this.renderer.renderMessages(session);
    }
    // Primary surfaces keep the legacy DOM around between tab switches,
    // so a plain re-activate doesn't re-render. But the browser resets
    // scroll position on any ancestor that went display:none (the
    // surface-container's data-focused flip), so when the user tabs
    // back to their chat they land at the top. Snap the primary
    // scrollEl back to the bottom on activate. The same "rAF twice"
    // dance covers layout settling after visibility flips on.
    if (this._isPrimary && typeof window !== 'undefined') {
      // Import lazily so this file doesn't need a direct dep on the
      // chat module's internal primary-renderer helper.
      import('../chat/index.js').then(mod => {
        mod.scrollPrimaryToBottom?.();
      }).catch(() => {});
    }
  }

  unmount() {
    if (this._primarySessionListener) {
      document.removeEventListener('augmentum:session-changed', this._primarySessionListener);
      this._primarySessionListener = null;
    }
    if (this._primaryModeListener) {
      document.removeEventListener('augmentum:mode-changed', this._primaryModeListener);
      this._primaryModeListener = null;
    }
    if (this._sessionChangedListener) {
      document.removeEventListener('augmentum:session-changed', this._sessionChangedListener);
      this._sessionChangedListener = null;
    }
    if (this._isPrimary) {
      // Return adopted singletons to main-area with the `hidden` class re-applied.
      // Without the re-hide, the elements stay `flex: 1` siblings of #surface-grid
      // and main-area splits 50/50 ("glitched splitscreen"). CoderSurface.unmount
      // hides its adopted DOM for the same reason — mirror that here so mode-
      // swap teardown is safe even when no new primary picks them up before paint.
      const mainArea = document.querySelector('.main-area');
      if (this._container) {
        const chatScroll = this._container.querySelector('#chat-scroll');
        const inputArea = this._container.querySelector('.input-area');
        if (mainArea && chatScroll) {
          chatScroll.classList.add('hidden');
          mainArea.appendChild(chatScroll);
        }
        if (mainArea && inputArea) {
          inputArea.classList.add('hidden');
          mainArea.appendChild(inputArea);
        }
      }
    } else {
      // Clean up independent DOM
      this.stream?.abort();
      this.renderer?.destroy();
      this.input?.destroy();
      this.renderer = null;
      this.input = null;
      this.stream = null;
    }
    super.unmount();
  }

  // ---------------------------------------------------------------------------
  // Send flow
  // ---------------------------------------------------------------------------

  async _handleSend(text, images, docs) {
    if (!text && (!images || images.length === 0) && (!docs || docs.length === 0)) return;
    if (this.stream?.isActive()) return;

    const session = sessionStore.get(this._sessionId);
    if (!session) return;

    this.input.setStreaming(true);

    // Add user node to session tree
    const parentId = session.activeLeafId || null;
    const userNode = tree.addChildNode(session, parentId, 'user', text);
    if (images && images.length > 0) {
      userNode.images = images;
    }
    session.activeLeafId = userNode.id;
    sessionStore.save(session.id);

    // Render user message
    this.renderer.renderMessages(session);

    // Create streaming message
    this.renderer.createStreamingMessage(session);

    // Stream response
    await this.stream.send(session, {
      model: app.state.currentModel || '',
      mode: this._mode,
      tools: this._getToolsList(),
      voiceInput: false,
    });
  }

  _handleStop() {
    this.stream?.abort();
    this.input?.setStreaming(false);
  }

  _handleMeta(meta) {
    if (!this.renderer) return;

    if (meta.status) {
      this.renderer.updateStreamingStatus(meta.status);
    }
    if (meta.phases) {
      this.renderer.updateStreamThinking(meta.phases, meta.complexity);
    }
    if (meta.phase_content_delta && meta.phase) {
      this.renderer.addStreamPhaseContent(meta.phase, meta.phase_content_delta);
    }
    if (meta.tool_call) {
      this.renderer.addStreamToolCall(meta.tool_call);
      if (meta.tool_call.phase === 'passthrough') {
        this.renderer.renderToolCallResult(meta.tool_call);
      }
      if (meta.tool_call.success !== false && meta.tool_call.result_metadata) {
        renderToolDeliverable(
          meta.tool_call.tool,
          meta.tool_call.result_metadata,
          { messageEl: this.renderer.streamingEl, toolCard: null },
        );
      }
    }
    if (meta.tool_status === 'running' && meta.tool_names) {
      this.renderer.showToolIndicator(meta.tool_names);
    }
    if (meta.model_thinking_delta) {
      this.renderer.appendModelThinking(meta.model_thinking_delta);
    }
    // Backend stage events (model_load / model_swap / slot_restore /
    // prefill). Mirrors the primary surface's handler in chat/index.js.
    if (meta.stage_start) {
      const s = meta.stage_start;
      const label = s.label || s.stage || '';
      const text = s.detail ? `${label} · ${s.detail}` : label;
      if (text) this.renderer.updateStreamingStatus(text);
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
      this.renderer.updateStreamMetrics({
        tokens_per_second: meta.tokens_per_second,
        context_length: meta.context_length,
        context_used: meta.context_used,
        prompt_tokens: meta.prompt_tokens,
        eval_tokens: meta.eval_tokens,
        ttft_ms: meta.ttft_ms,
        total_duration_ms: meta.total_duration_ms,
        eval_duration_ms: meta.eval_duration_ms,
      });
    }
    if (meta.flow_name) {
      this.renderer.setStreamFlowName(meta.flow_name);
    }
    if (meta.regex_transformed) {
      this.renderer.replaceStreamedContent(meta.regex_transformed);
    }
  }

  _handleStreamComplete(result) {
    if (!this.renderer || !this.input) return;
    this.input.setStreaming(false);

    // Continue-in-place: capture + clear the flag up front so it can never
    // leak into the next turn (even on abort / early return).
    const continuingNodeId = this._continuingNodeId;
    this._continuingNodeId = null;

    if (result.aborted) {
      this.renderer.finalizeStreaming();
      return;
    }

    const session = sessionStore.get(this._sessionId);
    if (!session) return;

    // Get streamed content and create assistant node
    const rawContent = this.renderer.getStreamingRawContent();

    // Continuation: the streamingEl IS the existing assistant node
    // (resumeStreamingMessage swapped it in). Write [partial + streamed] back
    // onto it — no new node, no parent attach, no activeLeafId move — mirroring
    // the primary surface's continue-complete branch.
    if (continuingNodeId) {
      const existingNode = session.tree?.[continuingNodeId];
      if (existingNode && existingNode.role === 'assistant') {
        if (rawContent) existingNode.content = rawContent;
        const reasoning = this.renderer.collectReasoningData();
        if (reasoning && reasoning.thinking) {
          const prior = (existingNode.reasoning && existingNode.reasoning.thinking) || '';
          existingNode.reasoning = {
            ...(existingNode.reasoning || {}),
            thinking: prior + reasoning.thinking,
          };
        }
        this.renderer.finalizeStreaming(session);
        sessionStore.save(session.id);
        return;
      }
      // Existing node lost — fall through to fresh-node creation.
    }
    if (rawContent) {
      const parentId = session.activeLeafId;
      const assistantNode = tree.addChildNode(session, parentId, 'assistant', rawContent);

      // Store reasoning data — UARF phases, model-native thinking, or both.
      // Thinking-only (no phases) still persists so reasoning models round-
      // trip across refreshes and server restarts.
      const reasoning = this.renderer.collectReasoningData();
      if (reasoning && ((reasoning.phases && reasoning.phases.length > 0) || reasoning.thinking)) {
        assistantNode.reasoning = reasoning;
      }

      // Store generation metrics
      const metrics = this.renderer._streamMetrics;
      if (metrics.tps > 0) assistantNode.tokens_per_second = metrics.tps;
      if (metrics.promptTokens > 0) assistantNode.prompt_tokens = metrics.promptTokens;
      if (metrics.evalTokens > 0) assistantNode.eval_tokens = metrics.evalTokens;
      if (metrics.contextLen > 0) {
        assistantNode.context_length = metrics.contextLen;
        assistantNode.context_used = metrics.contextUsed;
      }
      if (metrics.ttftMs > 0) assistantNode.ttft_ms = metrics.ttftMs;
      if (metrics.totalDurationMs > 0) assistantNode.total_duration_ms = metrics.totalDurationMs;
      if (metrics.evalDurationMs > 0) assistantNode.eval_duration_ms = metrics.evalDurationMs;

      session.activeLeafId = assistantNode.id;
      this.renderer.setStreamingNodeId(assistantNode.id);
    }

    // Pass session so finalizeStreaming can add the branch-swipe indicator
    // when this node has siblings (e.g. after a regenerate).
    this.renderer.finalizeStreaming(session);
    sessionStore.save(session.id);

    // Auto-generate title
    this._maybeGenerateTitle(session);
  }

  async _maybeGenerateTitle(session) {
    if (!session || session.title !== 'New Chat') return;
    if (!session.rootId) return;
    const root = session.tree[session.rootId];
    if (!root || root.role !== 'user') return;

    // Immediate fallback
    const text = root.content;
    session.title = text.slice(0, 50) + (text.length > 50 ? '...' : '');
    this.emit('surface:titleChanged', { title: session.title });

    // Async LLM title
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
        this.emit('surface:titleChanged', { title: session.title });
      }
    } catch { /* truncated title is fine */ }
  }

  // ---------------------------------------------------------------------------
  // Message actions
  // ---------------------------------------------------------------------------

  _handleAction(action, nodeId, data) {
    const session = sessionStore.get(this._sessionId);
    if (!session) return;

    switch (action) {
      case 'branch': {
        tree.switchToSibling(session, nodeId, data);
        this.renderer.renderMessages(session);
        sessionStore.save(session.id);
        break;
      }
      case 'regenerate': {
        this._regenerateMessage(session, nodeId);
        break;
      }
      case 'continue': {
        this._continueMessage(session, nodeId);
        break;
      }
      case 'edit': {
        // Shared inline-edit UI, owned by this surface's own renderer.
        const node = session.tree[nodeId];
        if (node) this.renderer.beginInlineEdit(nodeId, node.content);
        break;
      }
      case 'cancel-edit': {
        this.renderer.cancelInlineEdit();
        break;
      }
      case 'save-edit': {
        const node = session.tree[nodeId];
        if (!node) break;
        if (node.role === 'assistant') {
          node.content = data.content;
          sessionStore.save(session.id);
          this.renderer.renderMessages(session);
        } else {
          // User edit → create branch and re-send
          const parentId = node.parentId;
          const newUserNode = tree.addChildNode(session, parentId, 'user', data.content);
          session.activeLeafId = newUserNode.id;
          sessionStore.save(session.id);
          this.renderer.renderMessages(session);
          // Re-stream
          this.renderer.createStreamingMessage(session);
          this.input.setStreaming(true);
          this.stream.send(session, {
            model: app.state.currentModel || '',
            mode: this._mode,
            tools: this._getToolsList(),
          });
        }
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
          session.activeLeafId = null;
        }
        sessionStore.save(session.id);
        this.renderer.renderMessages(session);
        break;
      }
      case 'tts': {
        const node = nodeId && session ? session.tree?.[nodeId] : null;
        ttsPlayMessage(data?.text, data?.button, { speakerName: node?.speakerName || '' });
        break;
      }
      case 'impersonate': {
        this._impersonateMessage(session);
        break;
      }
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
        }
        break;
      }
      case 'send': {
        if (data?.text) this._handleSend(data.text, [], []);
        break;
      }
      case 'vote': {
        // A/B test vote
        if (data?.balancerId && data?.vote) {
          fetch(`/api/balancers/${data.balancerId}/vote`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              vote: data.vote,
              model: data.model,
              backend: data.backend,
            }),
          }).catch(() => {});
          // Persist vote on node
          const node = session.tree[nodeId];
          if (node) {
            node.ab_vote = data.vote;
            sessionStore.save(session.id);
          }
        }
        break;
      }
    }
  }

  async _regenerateMessage(session, nodeId) {
    const node = session.tree[nodeId];
    if (!node || node.role !== 'assistant') return;

    // Set active leaf to the user message before this one
    session.activeLeafId = node.parentId;
    sessionStore.save(session.id);
    this.renderer.renderMessages(session);

    // Create new streaming message and re-send
    this.renderer.createStreamingMessage(session);
    this.input.setStreaming(true);
    await this.stream.send(session, {
      model: app.state.currentModel || '',
      mode: this._mode,
      tools: this._getToolsList(),
    });
  }

  async _impersonateMessage(session) {
    if (this.stream?.isActive()) return;
    const persona = await fetchActivePersona();
    await runImpersonate(session, {
      mode: this._mode,
      sessionId: this._sessionId,
      userName: persona?.name || 'User',
      charName: session.title || 'the character',
      personaDesc: persona?.description || '',
      onStart: () => {
        this.input.setStreaming(true);
        this.input.setValue('');
      },
      onText: (acc) => this.input.setValue(acc),
      onEnd: () => {
        this.input.setStreaming(false);
        this.input.focus();
      },
    });
  }

  async _continueMessage(session, nodeId) {
    if (this.stream?.isActive()) return;
    // Continue = extend the trailing assistant message IN PLACE — matches the
    // primary surface (chat/index.js). No fake [Continue] user node, no new
    // assistant node; the backend continues the partial verbatim.
    const lastNode = session.tree?.[session.activeLeafId];
    if (!lastNode || lastNode.role !== 'assistant') return;
    if (lastNode.interrupted) {
      delete lastNode.interrupted;
      delete lastNode.error_message;
      delete lastNode.error_kind;
    }
    sessionStore.save(session.id);

    const resumed = this.renderer.resumeStreamingMessage(session.activeLeafId);
    if (!resumed) this.renderer.createStreamingMessage(session);
    this._continuingNodeId = session.activeLeafId;
    this.input.setStreaming(true);
    await this.stream.send(session, {
      model: app.state.currentModel || '',
      mode: this._mode,
      tools: this._getToolsList(),
      continueLastAssistant: true,
    });
  }

  _getToolsList() {
    if (this._mode === 'narrative') return 'none';
    const settings = getSettings();
    if (settings.passthroughAutoTools === false) return 'none';
    return (app.state.passthroughTools || []).join(',');
  }

  // ---------------------------------------------------------------------------
  // Surface interface
  // ---------------------------------------------------------------------------

  getTitle() {
    const modeTitle = { analytical: 'Analyze', agentic: 'Build', passthrough: 'Chat' };
    const fallback = modeTitle[this._mode] || 'Chat';
    if (this._isPrimary) return fallback;
    const session = sessionStore.get(this._sessionId);
    return session?.title === 'New Chat' ? fallback : (session?.title || fallback);
  }

  // Primary chat surface adopts #chat-scroll + .input-area from main-area;
  // closing it puts those elements back as siblings of #surface-grid, and
  // both compete for flex space → "glitched splitscreen". Hide close on
  // primary until the singleton-adoption migration is finished.
  isCloseable() { return !this._isPrimary; }

  getIcon() {
    // Must match CSS var names: --orb-passthrough, --orb-analytical, --orb-agentic
    return this._mode || 'passthrough';
  }

  getContext() {
    return {
      type: 'chat',
      id: this.id,
      mode: this._mode,
      sessionId: this._sessionId,
      summary: this._isPrimary ? 'Primary chat' : `Chat: ${this.getTitle()}`,
      capabilities: ['text', 'tools', 'search', 'calculate', 'build'],
    };
  }

  getState() {
    return {
      ...super.getState(),
      mode: this._mode,
      sessionId: this._sessionId,
      primary: this._isPrimary,
    };
  }

  restoreState(state) {
    super.restoreState(state);
    if (state.mode) this._mode = state.mode;
    if (state.sessionId) this._sessionId = state.sessionId;
    if (state.primary !== undefined) this._isPrimary = state.primary;
  }
}

// Auto-register
SurfaceRegistry.register('chat', ChatSurface);
