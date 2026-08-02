/* ==========================================================================
   CoderInstance — per-surface coder runtime state.

   Phase 1 of the multi-instance-surfaces design
   (docs/superpowers/specs/2026-08-01-multi-instance-surfaces-design.md).

   Today coder.js keeps its entire runtime as ~50 MODULE-GLOBALS, so there is
   exactly one coder view — which is why two coder tabs merge into one shared
   DOM (CoderSurface owns no per-instance DOM; see coder-surface.js). This
   class is the home those globals migrate into: one CoderInstance per
   CoderSurface, so N coder tabs become N independent workspaces (the backend
   already runs one-per-workspace concurrently, coder_background_run.py:39).

   Phase 1 is a deliberate NO-OP: a single module-level instance
   (`currentCoder()`) holds the state, and coder.js is migrated
   subsystem-by-subsystem off the globals onto it — behaviour identical, one
   instance. Phases 2-3 template the #coder-* DOM per surface and instance the
   subsystems (terminal/editor/stream/…), at which point `currentCoder()`
   becomes "the focused surface's instance" and merging is structurally
   impossible.

   Grouped to mirror the 17-surface inventory in the spec. Fields default to
   the exact initializers the coder.js globals had, so migrating a reference
   (`_x` → `inst.x`) is a pure textual move with no behaviour change. Each
   group carries a MIGRATED flag comment so the incremental sweep is auditable.
   ========================================================================== */

export class CoderInstance {
  constructor() {
    // ── Identity / workspace binding (root singleton state) ──────────────
    this.workspaceId = null;            // ex-_activeWorkspaceId
    this.terminalId = null;             // ex-_activeTerminalId
    this.status = 'stopped';            // ex-_activeStatus (project_checkouts.status)
    this.safeguardsEnabled = true;      // ex-_activeSafeguardsEnabled
    this.alwaysOn = false;              // ex-_activeAlwaysOn
    this.lanAccessible = false;         // ex-_activeLanAccessible
    this.agentSet = null;               // Phase 4: primary chat model/agent captured at open

    // ── Runs / external agents (augmentum · claude · pi) ─────────────────
    this.activeRunId = '';              // ex-_activeRunId
    this.claudeConnected = false;       // ex-_claudeConnected
    this.claudeResumeRunId = '';        // ex-_claudeResumeRunId
    this.claudeActiveRunId = '';        // ex-_claudeActiveRunId
    this.unifiedRunsById = new Map();   // ex-_unifiedRunsById
    this.agentDetailStream = null;      // ex-_agentDetailStream (CoderStream)

    // ── Editor (CodeMirror, id-keyed in cm-editor.js) ────────────────────
    this.activeEditorId = null;         // ex-_activeEditorId
    this.activeFilePath = '';           // ex-_activeFilePath
    this.editorFiles = [];              // ex-_editorFiles [{path,name,editorId}]
    this.activeEditorFile = null;       // ex-_activeEditorFile

    // ── Preview (iframe · ports · trust · workbench tab) ─────────────────
    this.previewInfo = {  // ex-_previewInfo
      state: 'not_published', published: false, ready: false,
      ready_count: 0, primary_url: null, urls: [],
    };
    this.previewPorts = [];             // ex-_previewPorts
    this.activePreviewUrl = '';         // ex-_activePreviewUrl
    this.activeWorkbenchTab = 'terminal';  // ex-_activeWorkbenchTab
    this.filePreview = null;            // ex-_filePreview { url, filePath, fileName }

    // ── Conversation / stream / mission (instance classes already) ───────
    this.conversation = null;           // ex-_conversation (CoderConversation)
    this.coderStream = null;            // ex-_coderStream (CoderStream)
    this.missionPanel = null;           // ex-_missionPanel (MissionPanel)
    this.activePromptDisposable = null; // ex-_activePromptDisposable
    this.chatHistory = [];              // ex-_chatHistory
    this.pendingAttachments = [];       // ex-_pendingAttachments
    this.terminalAgentAbort = null;     // ex-_terminalAgentAbort
    this.lastAgentRequest = null;       // ex-_lastAgentRequest { request, attachments }

    // ── Code intelligence (CodeMind) ── MIGRATED (Phase 1, slice 1) ──────
    this.codeMindReady = false;         // ex-_codeMindReady
    this.codeMindInitPromise = null;    // ex-_codeMindInitPromise
    this.codeMindDebounce = null;       // ex-_codeMindDebounce
    this.activeEditorCodeMindLanguage = null;  // ex-_activeEditorCodeMindLanguage
    this.activeEditorDiagnostics = [];  // ex-_activeEditorDiagnostics
    this.activeEditorDiagnosticsToken = 0;     // ex-_activeEditorDiagnosticsToken

    // ── Runtime misc ─────────────────────────────────────────────────────
    this.runtimePowerActivation = null; // ex-_runtimePowerActivation
    this.runDetailsDrawer = null;       // ex-_runDetailsDrawer
    this.reviewListenerWired = false;   // ex-_reviewListenerWired

    // ── Per-instance pollers/timers (Phase 3: one set per surface) ───────
    this.gitPollInterval = null;        // ex-_gitPollInterval
    this.portsPollInterval = null;      // ex-_portsPollInterval
    this.workspaceStatusPoll = null;    // ex-_workspaceStatusPoll
    this.fileTreeRefreshTimer = null;   // ex-_fileTreeRefreshTimer

    // ── DOM cache (Phase 2: per-surface templated #coder-* cluster) ──────
    this.dom = {};                      // ex-_dom
  }
}

// Phase 1 single active instance. In Phase 2 this becomes a lookup of the
// FOCUSED surface's instance; every migrated `currentCoder().x` reference then
// automatically resolves per-surface with no further edits.
let _active = new CoderInstance();

/** The coder instance driving the current view. */
export function currentCoder() {
  return _active;
}

/** Replace the active instance (Phase 2: on surface focus). Phase 1: tests only. */
export function setCurrentCoder(inst) {
  _active = inst;
  return _active;
}
