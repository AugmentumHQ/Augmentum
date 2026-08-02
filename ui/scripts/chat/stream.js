/* ==========================================================================
   Chat Module — ChatStream
   NDJSON streaming for a single ChatSurface.
   Each surface owns its own ChatStream instance.
   ========================================================================== */

import { extractErrorMessage } from '../app.js';
import { getSettings, buildPersonalizationPrompt, getThinkingOverrideForModel, getReasoningEffortForModel, stripPromptScaffolding } from '../settings.js';
import { buildSamplingOptions } from './constants.js';
import { buildMessagesForAPI } from './tree.js';
import { getCurrentFlow, getTuneOverrides } from '../flow-bar.js';

export class ChatStream {
  /**
   * @param {Object} options
   * @param {(text: string) => void}   options.onContent      — content delta
   * @param {(meta: Object) => void}   options.onMeta         — augmentum metadata
   * @param {(result: Object) => void} options.onComplete     — stream finished
   * @param {(error: Error) => void}   options.onError        — non-abort error
   * @param {(label: string) => void}  options.onStatusUpdate — status label change
   * @param {Object|null}              options.surface        — owning surface;
   *        non-primary surfaces skip the GLOBAL flow-bar's flow/tune headers
   *        (see the flow header block in send())
   */
  constructor(options = {}) {
    this.surface = options.surface || null;
    this._abortController = null;
    this._onContent = options.onContent || (() => {});
    this._onMeta = options.onMeta || (() => {});
    this._onComplete = options.onComplete || (() => {});
    this._onError = options.onError || (() => {});
    this._onStatusUpdate = options.onStatusUpdate || (() => {});
    // Stall watchdogs — content-based, not chunk-based. The server's ~3s
    // heartbeats deliberately do NOT reset these, so a backend that's
    // alive but stuck still surfaces a stall. Two thresholds:
    //   - thinking (4s):  soft hint when no content has arrived yet
    //   - stalled  (30s): friendly "Try again" affordance — the
    //                     dispatch stage from _with_heartbeat suspends
    //                     this counter for normal prep/load windows,
    //                     so reaching 30s here genuinely means the
    //                     content stream is silent past every known
    //                     slow phase. Was 15s before the dispatch-stage
    //                     handoff landed and tripped a panic banner
    //                     during routine model loads.
    this._thinkingDelayMs = 4000;
    this._stallTimeoutMs = 30000;
    this._thinkingTimer = null;
    this._stallTimer = null;
    this._isThinking = false;
    this._isStalled = false;
    // Active backend stages (model_load / model_swap / slot_restore / prefill).
    // While at least one is in flight, watchdogs are suspended — the backend
    // has explicitly declared a known-slow phase and the status label
    // already shows what's happening ("Loading model · deepseek-v3-instruct").
    // Showing "Stream stalled" with an Abort & retry button during a normal
    // 30-60s model load just panics users into cancelling work that would
    // have completed fine. Keyed by the stage's stable id so overlapping /
    // back-to-back stages compose cleanly.
    this._activeStages = new Map();
    // Safety net: if a stage hangs without ever emitting stage_complete
    // (genuinely stuck, not just slow), surface a stall at 5 min so the
    // user still gets an escape hatch.
    this._stageMaxBudgetMs = 5 * 60 * 1000;
  }

  /** Reset both watchdogs after content arrival (or at stream start).
   *  Content is the all-clear signal — clears any active hint/banner,
   *  then re-arms the countdown so a subsequent stall is caught.
   *
   *  No-op while a backend stage is in flight: the status label already
   *  surfaces "Loading model · X" / "Prefilling · N tokens", and a 30-60s
   *  load is normal, not a stall. Watchdogs resume when the last stage
   *  completes (see _noteStageComplete).
   */
  _armWatchdogs() {
    if (this._thinkingTimer) clearTimeout(this._thinkingTimer);
    if (this._stallTimer) clearTimeout(this._stallTimer);
    if (this._isThinking) {
      this._isThinking = false;
      this._onMeta({ thinking: false });
    }
    if (this._isStalled) {
      this._isStalled = false;
      this._onMeta({ stalled: false });
    }
    if (this._activeStages.size > 0) return;
    this._thinkingTimer = setTimeout(() => {
      this._isThinking = true;
      this._onMeta({ thinking: true });
    }, this._thinkingDelayMs);
    this._stallTimer = setTimeout(() => {
      this._isStalled = true;
      this._onMeta({ stalled: true });
    }, this._stallTimeoutMs);
  }

  /** Suspend watchdogs and start tracking a backend stage.
   *  The stage's own absolute-cap timer is the only thing that can
   *  surface a stall while the stage is active. */
  _noteStageStart(stageEvent) {
    if (!stageEvent || !stageEvent.id) return;
    const wasEmpty = this._activeStages.size === 0;
    const capTimer = setTimeout(() => {
      this._activeStages.delete(stageEvent.id);
      if (this._activeStages.size === 0) {
        this._isStalled = true;
        this._onMeta({ stalled: true });
      }
    }, this._stageMaxBudgetMs);
    this._activeStages.set(stageEvent.id, capTimer);
    if (wasEmpty) {
      if (this._thinkingTimer) {
        clearTimeout(this._thinkingTimer);
        this._thinkingTimer = null;
      }
      if (this._stallTimer) {
        clearTimeout(this._stallTimer);
        this._stallTimer = null;
      }
      if (this._isThinking) {
        this._isThinking = false;
        this._onMeta({ thinking: false });
      }
      if (this._isStalled) {
        this._isStalled = false;
        this._onMeta({ stalled: false });
      }
    }
  }

  /** Finish tracking a backend stage. Re-arms the content watchdogs
   *  once the last in-flight stage closes — back to the normal
   *  "15s without content = stalled" regime for the actual generation. */
  _noteStageComplete(stageEvent) {
    if (!stageEvent || !stageEvent.id) return;
    const capTimer = this._activeStages.get(stageEvent.id);
    if (capTimer) clearTimeout(capTimer);
    this._activeStages.delete(stageEvent.id);
    if (this._activeStages.size === 0 && this._abortController) {
      this._armWatchdogs();
    }
  }

  _clearWatchdogs() {
    if (this._thinkingTimer) {
      clearTimeout(this._thinkingTimer);
      this._thinkingTimer = null;
    }
    if (this._stallTimer) {
      clearTimeout(this._stallTimer);
      this._stallTimer = null;
    }
    for (const capTimer of this._activeStages.values()) {
      clearTimeout(capTimer);
    }
    this._activeStages.clear();
    // Emit clears so any UI banner / send-btn class consumer can
    // reset without each one polling. Cheap idempotence — if the
    // flag wasn't set, the consumer sees the same value twice.
    if (this._isThinking) {
      this._isThinking = false;
      this._onMeta({ thinking: false });
    }
    if (this._isStalled) {
      this._isStalled = false;
      this._onMeta({ stalled: false });
    }
  }

  /**
   * Start streaming a chat request.
   *
   * @param {Object} session — full session object (tree, id, mode, characterId, etc.)
   * @param {Object} options
   * @param {string}  options.model      — model name (default '')
   * @param {string}  options.mode       — override mode (default session.mode || 'passthrough')
   * @param {string}  options.tools      — comma-separated tool list
   * @param {boolean} options.voiceInput — flag STT-sourced input
   * @param {boolean|null} options.think — explicit thinking override
   */
  async send(session, options = {}) {
    // Abort any in-flight stream on this surface
    this.abort();

    this._abortController = new AbortController();
    const { signal } = this._abortController;

    // Declared at send() scope so the finally block can see it — JS `let`
    // inside try/finally would not be visible to the finally branch.
    let _pinnedThisTurn = '';

    try {
      // ---- Build request body ------------------------------------------------
      const settings = getSettings();
      const mode = options.mode || session.mode || 'passthrough';
      const chatMessages = buildMessagesForAPI(session);
      const samplingOptions = buildSamplingOptions(settings);
      // Per-chat sampling override. Stored on the session so it travels with
      // the conversation; only the keys set for THIS chat override the global
      // sampling. Flows to the backend inside `options`, which the apply-point
      // (models/sampling_profiles.py) treats as the highest-precedence
      // per-call layer — above the per-model profile and family default.
      if (session.chatSampling && typeof session.chatSampling === 'object') {
        Object.assign(samplingOptions, session.chatSampling);
      }

      // Narrative mode stores its card under narrativeSystemPrompt; passthrough
      // uses systemPrompt. Prefer narrative when present. Strip `// `
      // scaffolding lines from style-template chip output — those are
      // user-facing pick-one comments, not instructions for the model.
      let systemContent = stripPromptScaffolding(
        session.narrativeSystemPrompt || session.systemPrompt || ''
      );

      // Personalization prompt for non-narrative modes (prepended)
      const personalization = buildPersonalizationPrompt(mode);
      if (personalization) {
        systemContent = systemContent
          ? `${personalization}\n\n${systemContent}`
          : personalization;
      }

      // Assemble the message array. The greeting stays as role:assistant
      // (ST convention — it's an in-scene action, not a directive). Example
      // dialogue, when present, is injected as a role:system message
      // immediately before the final user message so it sits in the
      // model's "just saw this" window — matches SillyTavern's
      // pin_examples=false default.
      const messages = [];
      if (systemContent) {
        messages.push({ role: 'system', content: systemContent });
      }

      const examples = session.narrativeExamples || '';
      if (examples && chatMessages.length > 0 && chatMessages[chatMessages.length - 1].role === 'user') {
        // Insert example dialogue right before the trailing user turn
        const history = chatMessages.slice(0, -1);
        const userTurn = chatMessages[chatMessages.length - 1];
        messages.push(...history);
        messages.push({ role: 'system', content: `[Example Dialogue]\n${examples}` });
        messages.push(userTurn);
      } else {
        messages.push(...chatMessages);
      }

      const requestBody = {
        model: options.model || '',
        messages,
        stream: true,
      };
      if (options.continueLastAssistant) {
        // Backend reads this and asks the provider to extend the
        // trailing assistant message verbatim (DeepSeek prefix:true,
        // Anthropic native, llama-server add_generation_prompt:false,
        // synthetic-user fallback for providers without prefix).
        requestBody.continue_last_assistant = true;
      }
      const thinkOverride = typeof options.think === 'boolean'
        ? options.think
        : getThinkingOverrideForModel(options.model || '');
      if (Object.keys(samplingOptions).length > 0) {
        requestBody.options = samplingOptions;
      }
      if (options.voiceInput) {
        requestBody.voice_input = true;
      }
      if (thinkOverride !== null) {
        requestBody.think = thinkOverride;
      }
      // Qwen 3.6 only consumes preserve_thinking; the UI's preserve popover
      // is gated to that family in detectThinkingSupport. We forward the
      // flag whenever thinking is on — the backend re-gates by family so
      // a non-Qwen3.6 model never gets the kwarg.
      if (thinkOverride === true && settings.preserveThinking) {
        requestBody.preserve_thinking = true;
      }
      // OpenAI-family per-turn reasoning effort. Sourced from the
      // composer dropdown (or settings.reasoningEffort fallback);
      // adapter layer gates so non-OpenAI providers never see it.
      // None = let the mode hint decide.
      const reasoningEffort =
           (typeof options.reasoningEffort === 'string' && options.reasoningEffort)
        // Settings fallback is gated per-model: only levels the current
        // model's family actually accepts are sent (OpenAI 5-level enum vs
        // DeepSeek high/max; 'off' rides the think flag, never the wire).
        || getReasoningEffortForModel(options.model || '')
        || '';
      if (reasoningEffort) {
        requestBody.reasoning_effort = reasoningEffort;
      }
      // Narrative-mode UI lorebook: ship the live entries so the backend
      // LoreEngine (recursion, secondary keys, depth scan, probability,
      // groups, timed effects, card-field scanning) runs against what the
      // user actually sees in the UI — not just what a character_book had
      // at session init.
      if (session.lorebook && session.lorebook.length > 0) {
        requestBody.lorebook = session.lorebook;
      }

      // ---- Build headers -----------------------------------------------------
      const headers = {
        'Content-Type': 'application/json',
        'X-Augmentum-Mode': mode,
        'X-Augmentum-Session': session.id || '',
        'X-Augmentum-Tools': options.tools || '',
      };
      if (session.characterId) {
        headers['X-Augmentum-Character'] = session.characterId;
      }
      if (session.groupId) {
        // Tells the narrative handler this session is a group chat.
        // The backend loads the group from the DB by id, builds a turn
        // manager, and swaps the speaker's card per turn.
        headers['X-Augmentum-Group-Id'] = session.groupId;
      }
      // Per-turn manual speaker pin (Speaker Bar click or /as slash command).
      // Wins over rotation/LLM-decide for this single turn. We do NOT clear
      // nextSpeaker here — the Bar keeps showing the pinned state while the
      // pinned character is streaming their response. Release happens after
      // the stream completes (see the finally block below).
      if (session.nextSpeaker) {
        headers['X-Augmentum-Speaker'] = session.nextSpeaker;
        _pinnedThisTurn = session.nextSpeaker;
      }
      if (thinkOverride !== null) {
        headers['X-Augmentum-Think'] = thinkOverride ? 'true' : 'false';
      }
      // Media-player ambient context — when the user is actively
      // listening to a book, give the LLM a single factual sentence
      // about what it is. Enables "what's the magic system called"
      // without the user having to say which book. No-op when no audio.
      try {
        const mp = await import('../media-player.js');
        const st = mp.getState();
        if (st.fileId) {
          headers['X-Augmentum-Media-Context'] = JSON.stringify({
            fileId:        st.fileId,
            title:         st.title,
            author:        st.author,
            chapterIdx:    st.currentChapterIdx,
            chapterTitle:  st.chapters[st.currentChapterIdx]?.title || '',
            currentTimeS:  st.currentTimeS,
            durationS:     st.durationS,
            isPlaying:     st.isPlaying,
          });
        }
      } catch { /* module not loaded yet — skip silently */ }
      // Reasoning-flow headers come from the SINGLETON flow-bar, which only
      // the primary composer hosts. A stream owned by a non-primary surface
      // must not pick them up: before this gate, selecting a flow in the
      // primary Analyze tab silently attached X-Augmentum-Flow (+ tune
      // overrides) to every secondary tab's requests with no UI on that tab
      // showing it (surface-owned composer spec, acceptance criterion 7).
      // Legacy callers construct ChatStream without a surface — they ARE the
      // primary path and keep the flow.
      const ownsGlobalFlowBar = !this.surface || this.surface._isPrimary;
      if (ownsGlobalFlowBar) {
        const activeFlow = getCurrentFlow();
        if (activeFlow && activeFlow.id) {
          headers['X-Augmentum-Flow'] = activeFlow.id;
        }
        const tuneOverrides = getTuneOverrides();
        if (tuneOverrides) {
          headers['X-Augmentum-Flow-Tune'] = JSON.stringify(tuneOverrides);
        }
      }

      // ---- Fetch + NDJSON reader ---------------------------------------------
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers,
        body: JSON.stringify(requestBody),
        signal,
      });

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(extractErrorMessage(errBody, `HTTP ${response.status}: ${response.statusText}`));
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      this._armWatchdogs();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        // Watchdogs are NOT reset here — that's the whole point of the
        // content-based design. Heartbeats keep TCP alive but don't
        // suppress the stall signal. Content arrival in _processChunk
        // is the only thing that re-arms the timers.

        buffer += decoder.decode(value, { stream: true });

        // Split on newlines, process each complete JSON line
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep incomplete trailing fragment

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            this._processChunk(data);
          } catch (err) {
            // Re-throw backend errors so the outer catch routes them
            // through onError. Without this guard, JSON.parse failures
            // (malformed lines from a partial chunk) would also bubble
            // and abort the stream — which is wrong, those happen
            // routinely on chunked-encoding boundaries.
            if (err?.isBackendError) throw err;
            /* otherwise: malformed line, skip */
          }
        }
      }

      // Process any remaining buffer content
      if (buffer.trim()) {
        try {
          const data = JSON.parse(buffer);
          this._processChunk(data);
        } catch (err) {
          if (err?.isBackendError) throw err;
          /* otherwise: malformed trailing fragment, ignore */
        }
      }

      this._onComplete({ aborted: false });
    } catch (err) {
      if (err.name === 'AbortError') {
        this._onComplete({ aborted: true });
      } else {
        this._onError(err);
      }
    } finally {
      this._abortController = null;
      this._clearWatchdogs();
      // Release a per-turn speaker pin only AFTER the response (or abort).
      // Keeping the pin visible during streaming matches user expectation:
      // "I pinned Alice, Alice is speaking, now it goes back to auto."
      // If the user re-pinned mid-stream, don't overwrite their new choice.
      if (_pinnedThisTurn && session.nextSpeaker === _pinnedThisTurn) {
        session.nextSpeaker = '';
        document.dispatchEvent(new CustomEvent('augmentum:speaker-pin-released', {
          detail: { sessionId: session.id },
        }));
      }
    }
  }

  /**
   * Process a single parsed NDJSON chunk.
   * Handles Ollama-format, OpenAI-format, and Augmentum metadata.
   * @param {Object} data — parsed JSON object
   */
  _processChunk(data) {
    // Backend-error chunk — emitted by streaming.py::_make_error_chunk
    // when an inference call timed out / OOM'd / returned 5xx. Carries
    // both the legacy inline `[Error: ...]` text (for non-Augmentum
    // Ollama clients) AND a structured augmentum.backend_error field.
    // We detect the structured field FIRST so the `[Error: ...]` text
    // doesn't render into the response bubble as if the model wrote it.
    // Throwing here lets the existing send() try/catch route this
    // through onError, which persists the partial as an interrupted
    // node so the user can hit regenerate.
    if (data.augmentum?.backend_error) {
      const err = new Error(data.augmentum.backend_error);
      err.kind = data.augmentum.error_kind || 'backend_error';
      err.retryable = !!data.augmentum.retryable;
      err.isBackendError = true;
      throw err;
    }

    // OpenAI chat-completions style content delta
    if (data.message?.content) {
      this._armWatchdogs();
      this._onContent(data.message.content);
    }

    // Ollama raw response format
    if (data.response) {
      this._armWatchdogs();
      this._onContent(data.response);
    }

    // Augmentum-specific metadata (phases, status, tool calls, etc.)
    if (data.augmentum) {
      // Stage events (model_load / model_swap / slot_restore / prefill) ARE
      // the backend's liveness signal during known-slow phases. Pause the
      // content watchdog on stage_start so a normal 30-60s model load
      // doesn't fire the alarming "Abort & retry" banner; resume on
      // stage_complete. Inspected before _onMeta forward so the UI sees
      // a consistent stalled=false state when the stage suppresses it.
      if (data.augmentum.stage_start) {
        this._noteStageStart(data.augmentum.stage_start);
      }
      if (data.augmentum.stage_complete) {
        this._noteStageComplete(data.augmentum.stage_complete);
      }
      this._onMeta(data.augmentum);

      // Surface status updates for UI indicators
      if (data.augmentum.status) {
        this._onStatusUpdate(data.augmentum.status);
      }
    }

    // Ollama final chunk with generation stats
    if (data.done === true) {
      const aug = data.augmentum || {};
      const result = {};
      if (data.eval_count) result.evalTokens = data.eval_count;
      if (data.prompt_eval_count) result.promptTokens = data.prompt_eval_count;
      if (data.eval_duration) {
        result.tps = Math.round(data.eval_count / (data.eval_duration / 1e9));
      }
      this._onMeta({
        tokens_per_second: aug.tokens_per_second ?? result.tps,
        prompt_tokens: aug.prompt_tokens ?? result.promptTokens,
        eval_tokens: aug.eval_tokens ?? result.evalTokens,
        // Estimated flags only present when the proxy stream wrapper
        // populated them (post-2026-05-28 backends). Older paths emit
        // nothing here, which the UI treats as "authoritative" (no ~
        // marker) — that's the right fallback because it matches the
        // pre-flag behavior of all existing rows.
        prompt_tokens_estimated: aug.prompt_tokens_estimated || false,
        eval_tokens_estimated: aug.eval_tokens_estimated || false,
        context_length: aug.context_length,
        context_used: aug.context_used,
        prompt_tokens_evaluated: aug.prompt_tokens_evaluated,
        prompt_tokens_cached: aug.prompt_tokens_cached,
        prompt_tokens_cache_write: aug.prompt_tokens_cache_write,
        // KV reuse-audit verdict from the engine (kv_reuse_audit join):
        // names WHY a turn cold-prefilled when reuse was possible.
        kv_reuse: aug.kv_reuse,
        kv_void_cause: aug.kv_void_cause,
        reasoning_tokens: aug.reasoning_tokens,
        ttft_ms: aug.ttft_ms,
        total_duration_ms: aug.total_duration_ms ?? (data.total_duration ? Math.round(data.total_duration / 1e6) : undefined),
        eval_duration_ms: aug.eval_duration_ms ?? (data.eval_duration ? Math.round(data.eval_duration / 1e6) : undefined),
        // Phase 8: model that served the turn, used by the renderer to
        // look up a peer-icon badge from the cached model list. Stays
        // empty when the backend didn't echo a model name (older paths).
        model_used: data.model || '',
      });
    }
  }

  /** Abort the in-flight stream, if any. */
  abort() {
    if (this._abortController) {
      this._abortController.abort();
      this._abortController = null;
    }
  }

  /** Whether a stream is currently in progress. */
  isActive() {
    return this._abortController !== null;
  }
}
