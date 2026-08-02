/**
 * coder-stream.js - SSE/NDJSON stream parser for coder agent conversations.
 */

/**
 * Per-run "highest event seq the UI has rendered" floor, persisted in
 * sessionStorage so a reattach — whether from a fresh CoderStream
 * instance (repeat tab-sleep) or a full page reload — can resume past
 * what was already drawn instead of replaying the whole ring buffer
 * from seq 0. Keyed by run id; cleared when a run terminates.
 */
const _RUN_SEQ_KEY = 'augmentum.coder.runSeq.';

export function persistRunSeq(runId, seq) {
  if (!runId || !(Number(seq) > 0)) return;
  try {
    sessionStorage.setItem(_RUN_SEQ_KEY + runId, String(Number(seq)));
  } catch {
    // sessionStorage unavailable / quota — reattach just replays more.
  }
}

export function readRunSeq(runId) {
  if (!runId) return 0;
  try {
    return Number(sessionStorage.getItem(_RUN_SEQ_KEY + runId)) || 0;
  } catch {
    return 0;
  }
}

export function clearRunSeq(runId) {
  if (!runId) return;
  try {
    sessionStorage.removeItem(_RUN_SEQ_KEY + runId);
  } catch {
    // best-effort cleanup
  }
}

/**
 * Strip tool-call JSON objects from text using brace-depth matching.
 * Handles nested braces, including code blocks inside JSON strings.
 *
 * @param {string} text
 * @returns {string}
 */
export function stripToolCallJSON(text) {
  if (!text.includes('"tool"')) return text;
  let result = '';
  let i = 0;
  while (i < text.length) {
    if (text[i] === '{' && text.substring(i, i + 10).match(/\{\s*"tool"/)) {
      let depth = 0;
      let j = i;
      let inString = false;
      let escaped = false;
      while (j < text.length) {
        const ch = text[j];
        if (escaped) {
          escaped = false;
          j += 1;
          continue;
        }
        if (ch === '\\' && inString) {
          escaped = true;
          j += 1;
          continue;
        }
        if (ch === '"') inString = !inString;
        if (!inString) {
          if (ch === '{') depth += 1;
          else if (ch === '}') {
            depth -= 1;
            if (depth === 0) {
              j += 1;
              break;
            }
          }
        }
        j += 1;
      }
      i = j;
    } else {
      result += text[i];
      i += 1;
    }
  }
  return result;
}

/**
 * @typedef {Object} CoderStreamCallbacks
 * @property {(text: string) => void} onContent
 * @property {(id: string, tool: string, input: Object) => void} onToolCall
 * @property {(id: string, result: Object) => void} onToolResult
 * @property {(text: string) => void} onShellOutput
 * @property {(stalled: boolean) => void} onStall
 * @property {(text: string) => void} onThinking
 * @property {(text: string) => void} onReasoning
 * @property {(step: number, total: number, desc: string) => void} onStepStart
 * @property {(strategy: string) => void} onStrategy
 * @property {(response: string) => void} onComplete
 * @property {(error: string) => void} onError
 * @property {(phase: string, status: string, aug: Object) => void} onStatus
 * @property {(payload: {type: 'start'|'complete'|'progress', stage: string, label?: string, detail?: string, success?: boolean, duration_ms?: number, percent?: number, message?: string, id?: string}) => void} onStage
 * @property {(runId: string, aug: Object) => void} onRunDetails
 * @property {(payload: Object) => void} onPowerActivated
 * @property {(mission: Array) => void} onMissionStarted
 * @property {(promise: Object) => void} onPromiseStarted
 * @property {(promise: Object) => void} onPromiseVerifying
 * @property {(promise: Object) => void} onPromiseFulfilled
 * @property {(promise: Object, reason: string) => void} onPromiseRetry
 * @property {(promise: Object, reason: string) => void} onPromiseRejected
 * @property {(promise: Object, children: Array) => void} onPromiseDecomposed
 * @property {(data: Object) => void} onMissionCompleted
 * @property {(data: Object) => void} onMissionFailed
 * @property {(info: Object) => void} onRateLimited
 * @property {(turnId: string) => void} onReviewPending
 */

// Max time the dispatch loop may hold the main thread before yielding so the
// browser can paint + handle input. A single network chunk can carry a BURST
// of events (3-4 tool_call/tool_result pairs), and rendering each tool card
// (diff body, previews, layout) synchronously back-to-back blocks paint for
// the whole batch — the "screen freezes for a few seconds when tool calls come
// in at once" jank. Processing in ~8ms slices with a yield between lets each
// card paint as it lands. Total work is unchanged; it's just no longer one
// unbroken task.
const _DISPATCH_BUDGET_MS = 8;

/**
 * Yield to the browser so it can paint and process input, then resume.
 * Prefers the Scheduler API (resumes promptly after the browser's work);
 * falls back to a macrotask (setTimeout) which still crosses a task boundary
 * so a paint can happen. The non-browser/test path resolves immediately.
 *
 * Exported: coder-conversation.js uses the same budget-slice-yield pattern
 * for its history backfill (one shared discipline — see _DISPATCH_BUDGET_MS).
 */
export function yieldToPaint() {
  return _yieldToPaint();
}

function _yieldToPaint() {
  if (typeof globalThis !== 'undefined'
      && globalThis.scheduler && typeof globalThis.scheduler.yield === 'function') {
    return globalThis.scheduler.yield();
  }
  if (typeof setTimeout === 'function') {
    return new Promise((r) => setTimeout(r, 0));
  }
  return Promise.resolve();
}

export class CoderStream {
  /** @param {CoderStreamCallbacks} callbacks */
  constructor(callbacks = {}) {
    this._cb = callbacks;
    this._abort = null;
    this._active = false;
    this._lastContentDelta = '';
    this._lastThinkingDelta = '';
    // ── Soft stall watchdog (single tier, friendly) ──────────────────────
    // The coder emits a dense event stream — status transitions, tool
    // calls, shell bytes — so ANY of those is a sign of life and re-arms
    // the timer (see _pokeWatchdog call sites in _processChunk). Only a
    // genuine silence (no event of any kind for _stallTimeoutMs while NOT
    // inside a known-slow backend stage) surfaces a soft "still working"
    // hint via onStall(true). Deliberately NOT an abort banner: the coder
    // auto-retries transient failures and there's nothing for the user to
    // cancel — this is reassurance, not an alarm (progress over abort).
    // Longer than chat's 30s because coder iterations are heavier and a
    // quiet LLM call between tool batches is normal.
    this._stallTimeoutMs = 45000;
    this._stallTimer = null;
    this._isStalled = false;
    // Backend stages in flight (model_load / model_swap / slot_restore /
    // prefill). While any is active the watchdog is suspended —
    // coder-progress.js already shows the load/prefill bar there, so a
    // 30-120s load is expected, not a stall. Keyed by the stage id so
    // overlapping / back-to-back stages compose cleanly.
    this._activeStages = new Map();
    // Safety net: a stage that never emits stage_complete (genuinely hung,
    // not just slow) still releases a stall hint after this absolute cap.
    this._stageMaxBudgetMs = 5 * 60 * 1000;
    /**
     * Highest ``augmentum.seq`` seen on the current stream. Read by
     * the host on reattach: pass it as ``sinceSeq`` to skip events
     * we already rendered before the disconnect. Reset on every
     * send()/attach() call.
     */
    this.lastSeq = 0;
    /**
     * Current run id (filled by the first chunk carrying
     * ``augmentum.run_id``). Exposed so the host can persist it to
     * sessionStorage for cross-reload reattach.
     */
    this.runId = '';
  }

  /**
   * Send a request to the coder agent and stream the response.
   *
   * @param {Object} opts
   * @param {string} opts.model
   * @param {Array} opts.messages
   * @param {string} opts.workspaceId
   * @param {Object} [opts.extraHeaders]
   * @returns {Promise<string>}
   */
  async send({ model, messages, workspaceId, extraHeaders = {}, chatTemplateKwargs = null }) {
    if (this._active) this.abort();

    this._abort = new AbortController();
    this._active = true;
    this._lastContentDelta = '';
    this._lastThinkingDelta = '';
    this.lastSeq = 0;
    this.runId = '';
    this._pokeWatchdog();  // arm the silence watchdog for this turn

    try {
      // ``chat_template_kwargs`` carries per-turn template overrides
      // (e.g. {enable_thinking: true/false}). When non-null it's
      // forwarded to the backend, which the coder handler's _act_native
      // reads at the iteration boundary to honor the user's toggle.
      // Omitted from the body when null so the request shape stays
      // identical to legacy clients that don't set it.
      const body = { model, messages, stream: true };
      if (chatTemplateKwargs && typeof chatTemplateKwargs === 'object') {
        body.chat_template_kwargs = chatTemplateKwargs;
      }
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Augmentum-Mode': 'coder',
          'X-Augmentum-Workspace': workspaceId,
          ...extraHeaders,
        },
        body: JSON.stringify(body),
        signal: this._abort.signal,
      });

      if (!resp.ok) {
        const errText = await resp.text().catch(() => resp.statusText);
        this._cb.onError?.(`Server error ${resp.status}: ${errText}`);
        return '';
      }
      return await this._pumpResponse(resp);
    } catch (err) {
      if (err.name === 'AbortError') {
        this._cb.onComplete?.('');
      } else {
        this._cb.onError?.(err.message || 'Stream failed');
      }
      return '';
    } finally {
      this._active = false;
      this._abort = null;
      this._clearWatchdog();
    }
  }

  /**
   * Reattach to an in-flight coder run.
   *
   * Used when a previous fetch died (mobile screen sleep, tab switch,
   * laptop closed) but the agent task on the server kept running.
   * The /stream endpoint replays anything past ``sinceSeq`` from the
   * broker's ring buffer + tails new chunks. If the run already
   * finished, the server emits a final_state chunk with the saved
   * assistant message so the UI still gets ``onComplete``.
   *
   * @param {Object} opts
   * @param {string} opts.runId
   * @param {number} [opts.sinceSeq]
   * @returns {Promise<string>}
   */
  async attach({ runId, sinceSeq = 0 }) {
    if (this._active) this.abort();

    this._abort = new AbortController();
    this._active = true;
    this._lastContentDelta = '';
    this._lastThinkingDelta = '';
    this.lastSeq = Number(sinceSeq) || 0;
    this.runId = runId || '';
    this._pokeWatchdog();  // arm the silence watchdog for the reattach

    try {
      const url = `/api/coder/runs/${encodeURIComponent(runId)}/stream?since=${sinceSeq}`;
      const resp = await fetch(url, {
        method: 'GET',
        signal: this._abort.signal,
      });
      if (!resp.ok) {
        const errText = await resp.text().catch(() => resp.statusText);
        this._cb.onError?.(`Reattach failed ${resp.status}: ${errText}`);
        return '';
      }
      return await this._pumpResponse(resp);
    } catch (err) {
      if (err.name === 'AbortError') {
        this._cb.onComplete?.('');
      } else {
        this._cb.onError?.(err.message || 'Reattach failed');
      }
      return '';
    } finally {
      this._active = false;
      this._abort = null;
      this._clearWatchdog();
    }
  }

  /**
   * Drain an NDJSON response body, dispatching each chunk via
   * ``_processChunk``. Shared between send() and attach() since the
   * wire format is identical.
   *
   * @private
   * @param {Response} resp
   * @returns {Promise<string>}
   */
  async _pumpResponse(resp) {
    let fullResponse = '';
    let activeToolId = null;
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      const lines = buf.split('\n');
      buf = lines.pop() || '';

      // Frame-budgeted dispatch: process events until we've held the main
      // thread for ~_DISPATCH_BUDGET_MS, then yield so the browser can paint
      // the tool cards rendered so far and stay responsive to input. Without
      // this, a chunk carrying a burst of tool calls renders all of them in
      // one synchronous task and the screen freezes until it finishes.
      let frameStart = performance.now();
      for (const line of lines) {
        if (!line.trim() || line === 'data: [DONE]') continue;
        const jsonStr = line.startsWith('data: ') ? line.slice(6) : line;
        try {
          const chunk = JSON.parse(jsonStr);
          const acceptedDelta = this._processChunk(chunk, activeToolId, (id) => {
            activeToolId = id;
          });
          if (acceptedDelta) fullResponse += acceptedDelta;
        } catch {
          // Partial JSON line; wait for the next read.
        }
        if (performance.now() - frameStart > _DISPATCH_BUDGET_MS) {
          await _yieldToPaint();
          // If the user aborted during the yield, stop draining buffered
          // events; the outer reader.read() then drives the normal abort path
          // (which fires onComplete), so we don't bypass that contract here.
          if (this._abort?.signal?.aborted) break;
          frameStart = performance.now();
        }
      }
    }

    // Clean end of stream (true completion or a reattach that landed on
    // the run's final_state) — the floor is no longer needed. A
    // disconnect throws out of the read loop before reaching here, so we
    // correctly keep the floor for the next reattach.
    clearRunSeq(this.runId);
    this._cb.onComplete?.(fullResponse);
    return fullResponse;
  }

  /**
   * Process a single parsed stream chunk.
   *
   * @private
   * @param {Object} chunk
   * @param {string | null} activeToolId
   * @param {(id: string | null) => void} setActiveToolId
   * @returns {string}
   */
  _processChunk(chunk, activeToolId, setActiveToolId) {
    const aug = chunk.augmentum || chunk.choices?.[0]?.delta?.augmentum;

    const rawDelta = chunk.message?.content
      || chunk.choices?.[0]?.delta?.content
      || '';
    const delta = this._dedupeDelta('content', rawDelta);

    if (delta) {
      if (activeToolId && aug?.status === 'shell_output') {
        this._cb.onShellOutput?.(delta);
      } else if (aug?.status === 'mission_log') {
        // Intentional no-op. The mission panel renders this state itself.
      } else if (aug?.phase === 'planning') {
        this._cb.onThinking?.(delta);
      } else {
        const clean = stripToolCallJSON(delta.replace(/<task_complete\/>/g, ''));
        if (clean.trim()) {
          this._cb.onContent?.(clean);
        }
      }
    }

    const rawThinking = chunk.augmentum?.model_thinking_delta
      || chunk.choices?.[0]?.delta?.thinking
      || '';
    const thinking = this._dedupeDelta('thinking', rawThinking);
    if (thinking) {
      // ``reasoning_delta`` chunks carry live chain-of-thought (coalesced
      // server-side, ~2-4/s) → the collapsible reasoning block. Everything
      // else on the thinking channel (plan text arriving in-channel,
      // passthrough/conversational relays) keeps the legacy thinking
      // bubble so plan rendering is untouched.
      if (aug?.status === 'reasoning_delta') {
        this._cb.onReasoning?.(thinking);
      } else {
        this._cb.onThinking?.(thinking);
      }
    }

    // Any content/thinking byte is a sign of life — re-arm the watchdog.
    if (delta || thinking) this._pokeWatchdog();

    if (!aug) return delta;

    if (typeof aug.seq === 'number' && aug.seq > this.lastSeq) {
      this.lastSeq = aug.seq;
    }
    if (aug.run_id) {
      if (!this.runId) this.runId = aug.run_id;
      this._cb.onRunDetails?.(aug.run_id, aug);
    }

    if (aug.phase && aug.status && aug.status !== 'reasoning_delta') {
      // reasoning_delta is excluded: it arrives several times per second
      // while the model thinks, and this block does a sessionStorage
      // write + status-pill DOM update per call — the per-chunk-work
      // class the streaming-efficiency pass removed. The 'thinking'
      // transition chunk (once per sub-state change) drives the pill.
      this._cb.onStatus?.(aug.phase, aug.status, aug);
      // Persist the rendered-seq floor at status boundaries — far less
      // frequent than per-token deltas, so this is a handful of
      // sessionStorage writes per turn, not thousands.
      persistRunSeq(this.runId, this.lastSeq);
      // A status transition is progress — re-arm the stall watchdog. (A
      // chunk that ALSO carries stage_start will suspend it just below,
      // since the stage handling runs after this block.)
      this._pokeWatchdog();
    }

    // Backend Stage lifecycle events — model_load / model_swap /
    // slot_restore / prefill. Sent as augmentum.stage_start /
    // stage_complete / stage_progress dicts (see status_bus.Stage). Each
    // carries an ``id`` so the renderer can pair start with complete
    // when two stages overlap (rare but possible: prefill on slot N
    // while model_load is finishing for slot M+1).
    if (aug.stage_start) {
      const s = aug.stage_start;
      this._cb.onStage?.({
        type: 'start',
        stage: s.stage || '',
        label: s.label || '',
        detail: s.detail || '',
        id: s.id || '',
      });
      // Suspend the silence watchdog while a known-slow stage runs.
      this._noteStageStart(s.id || '');
    }
    if (aug.stage_complete) {
      const c = aug.stage_complete;
      this._cb.onStage?.({
        type: 'complete',
        stage: c.stage || '',
        success: c.success !== false,
        duration_ms: c.duration_ms || 0,
        detail: c.detail || '',
        id: c.id || '',
      });
      // Stage closed — re-arm the watchdog for the generation that follows.
      this._noteStageComplete(c.id || '');
    }
    if (aug.stage_progress) {
      const p = aug.stage_progress;
      this._cb.onStage?.({
        type: 'progress',
        stage: p.stage || '',
        percent: typeof p.percent === 'number' ? p.percent : undefined,
        message: p.message || '',
        id: p.id || '',
      });
    }
    if (aug.status === 'power_activated' && aug.power_activation) {
      this._cb.onPowerActivated?.(aug.power_activation);
    }

    if (aug.strategy) {
      this._cb.onStrategy?.(aug.strategy);
    }

    if (aug.status === 'tool_call' && aug.tool_call) {
      const tc = aug.tool_call;
      setActiveToolId(tc.id);
      this._cb.onToolCall?.(tc.id, tc.tool || tc.name, tc.input || {});
    }

    if (aug.status === 'tool_result' && aug.tool_result) {
      setActiveToolId(null);
      this._cb.onToolResult?.(aug.tool_result.id, aug.tool_result);
    }

    // Subagent live activity — one event per inner-loop boundary.
    // Carries an instance_id (minted by the dispatcher); the UI binds
    // it to the most-recent running task_dispatch card on first
    // sighting. See coder-conversation.js::updateSubagentProgress.
    if (aug.status === 'subagent_progress' && aug.subagent_progress) {
      this._cb.onSubagentProgress?.(aug.subagent_progress);
    }

    if (aug.status === 'step_start') {
      this._cb.onStepStart?.(aug.step, aug.total, aug.description || '');
    }

    if (aug.status === 'complete' && aug.review_turn_id) {
      this._cb.onReviewPending?.(aug.review_turn_id);
    }

    switch (aug.status) {
      case 'mission_started':
        if (Array.isArray(aug.mission)) this._cb.onMissionStarted?.(aug.mission);
        break;
      case 'promise_started':
        if (aug.promise) this._cb.onPromiseStarted?.(aug.promise);
        break;
      case 'promise_verifying':
        if (aug.promise) this._cb.onPromiseVerifying?.(aug.promise);
        break;
      case 'promise_fulfilled':
        if (aug.promise) this._cb.onPromiseFulfilled?.(aug.promise);
        break;
      case 'promise_retry':
        if (aug.promise) this._cb.onPromiseRetry?.(aug.promise, aug.reason || '');
        break;
      case 'promise_rejected':
        if (aug.promise) this._cb.onPromiseRejected?.(aug.promise, aug.reason || '');
        break;
      case 'promise_decomposed':
        if (aug.promise) {
          this._cb.onPromiseDecomposed?.(aug.promise, aug.children || []);
        }
        break;
      case 'mission_completed':
        this._cb.onMissionCompleted?.({ ...aug });
        break;
      case 'mission_failed':
        this._cb.onMissionFailed?.({ ...aug });
        break;
      case 'rate_limited':
        this._cb.onRateLimited?.({
          promise: aug.promise,
          waitSeconds: aug.wait_seconds || 0,
          attempt: aug.attempt || 1,
          maxRetries: aug.max_retries || 3,
          reason: aug.reason || '',
        });
        break;
    }

    return delta;
  }

  /** Re-arm (or first-arm) the stall watchdog. Called on every sign of
   *  life. Clears any standing "stalled" hint, then restarts the silence
   *  countdown. No-op while a backend stage is in flight — the load /
   *  prefill bar already explains the wait. */
  _pokeWatchdog() {
    if (this._stallTimer) clearTimeout(this._stallTimer);
    if (this._isStalled) {
      this._isStalled = false;
      this._cb.onStall?.(false);
    }
    if (this._activeStages.size > 0) return;  // suspended during a stage
    if (!this._active) return;
    this._stallTimer = setTimeout(() => {
      this._isStalled = true;
      this._cb.onStall?.(true);
    }, this._stallTimeoutMs);
  }

  /** A backend stage opened (model_load / prefill / …). Suspend the
   *  content watchdog; the stage's own absolute-cap timer is the only
   *  thing that can surface a stall while it runs. */
  _noteStageStart(id) {
    if (!id) return;
    const capTimer = setTimeout(() => {
      this._activeStages.delete(id);
      if (this._activeStages.size === 0 && this._active) {
        this._isStalled = true;
        this._cb.onStall?.(true);
      }
    }, this._stageMaxBudgetMs);
    this._activeStages.set(id, capTimer);
    if (this._stallTimer) {
      clearTimeout(this._stallTimer);
      this._stallTimer = null;
    }
    if (this._isStalled) {
      this._isStalled = false;
      this._cb.onStall?.(false);
    }
  }

  /** A backend stage closed. Re-arm the content watchdog once the last
   *  in-flight stage is done. */
  _noteStageComplete(id) {
    if (!id) return;
    const capTimer = this._activeStages.get(id);
    if (capTimer) clearTimeout(capTimer);
    this._activeStages.delete(id);
    if (this._activeStages.size === 0) this._pokeWatchdog();
  }

  /** Tear the watchdog down at stream end / abort / error. Idempotent. */
  _clearWatchdog() {
    if (this._stallTimer) {
      clearTimeout(this._stallTimer);
      this._stallTimer = null;
    }
    for (const capTimer of this._activeStages.values()) clearTimeout(capTimer);
    this._activeStages.clear();
    if (this._isStalled) {
      this._isStalled = false;
      this._cb.onStall?.(false);
    }
  }

  _dedupeDelta(kind, text) {
    if (!text) return '';
    const key = kind === 'thinking' ? '_lastThinkingDelta' : '_lastContentDelta';
    const duplicate = (
      this[key] === text
      && (text.includes('\n') || text.trim().length >= 80)
    );
    this[key] = text;
    return duplicate ? '' : text;
  }

  abort() {
    if (this._abort) {
      this._abort.abort();
      this._active = false;
    }
    this._clearWatchdog();
  }

  isActive() {
    return this._active;
  }
}
