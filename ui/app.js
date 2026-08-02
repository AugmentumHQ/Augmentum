/* ==========================================================================
   Augmentum Web UI — Application Logic
   ========================================================================== */

(function () {
  "use strict";

  // ---- Constants ----

  const STORAGE_SESSIONS = "augmentum_sessions";
  const STORAGE_SETTINGS = "augmentum_settings";
  const STORAGE_ACTIVE = "augmentum_active_session";

  const ANALYTICAL_PHASES = [
    "ASSESS",
    "IDENTIFY",
    "RELEVANT",
    "APPLY",
    "VERIFY",
    "CONCLUDE",
  ];

  // User-friendly display names for backend phase identifiers
  const PHASE_DISPLAY_NAMES = {
    "ASSESS": "Assess",
    "SEARCH": "Search",
    "GATHER": "Gather",
    "IDENTIFY": "Identify",
    "RELEVANT": "Research",
    "APPLY": "Analyze",
    "VERIFY": "Verify",
    "RESPOND": "Respond",
    "CONCLUDE": "Conclude",
  };

  // ---- State ----

  let sessions = {};        // { id: { id, title, version:2, tree:{nodeId: node}, rootId, activeLeafId, createdAt } }
  let activeSessionId = null;
  let selectedModel = "default";
  let currentMode = "passthrough";
  let isStreaming = false;
  let abortController = null;
  let pullAbortController = null;  // For model download cancellation
  let hasExternalNarrativeState = false; // Track external session detection

  // Flow editor state
  let flowList = [];                   // [{id, name, is_default, is_builtin, step_count, ...}]
  let currentFlow = null;              // Full flow object with steps
  let editingStepIndex = -1;           // Index of step being edited
  let editingStepOriginal = null;      // Original step data for revert
  let flowEditorMode = "editor";       // "editor" | "live"
  const KNOWN_TOOLS = [
    "web_search", "web_fetch", "python_exec", "calculator",
    "math_verify", "consistency_check", "memory_recall",
    "file_read", "file_write", "text_analysis", "json_tool",
    "hash_tool", "unit_converter",
  ];
  const TOOL_CATEGORIES = ["search", "fetch", "execute", "verify", "file"];

  // ---- Conversation Tree Helpers ----

  function generateNodeId() {
    return "n_" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  }

  function getPathToRoot(session, nodeId) {
    const path = [];
    let current = nodeId;
    while (current) {
      const node = session.tree[current];
      if (!node) break;
      path.unshift(current);
      current = node.parentId;
    }
    return path;
  }

  function getActivePath(session) {
    if (!session || !session.tree || !session.activeLeafId) return [];
    const ids = getPathToRoot(session, session.activeLeafId);
    return ids.map((id) => session.tree[id]).filter(Boolean);
  }

  function buildMessagesForAPI(session) {
    return getActivePath(session)
      .filter((node) => node.role !== "image")
      .map((node) => ({ role: node.role, content: node.content }));
  }

  function addChildNode(session, parentId, role, content) {
    const id = generateNodeId();
    const node = {
      id,
      role,
      content,
      parentId: parentId || null,
      children: [],
      createdAt: Date.now(),
    };
    session.tree[id] = node;
    if (parentId && session.tree[parentId]) {
      session.tree[parentId].children.push(id);
    }
    if (!parentId) {
      session.rootId = id;
    }
    return node;
  }

  function getSiblingInfo(session, nodeId) {
    const node = session.tree[nodeId];
    if (!node) return { siblings: [nodeId], index: 0, total: 1 };
    if (!node.parentId || !session.tree[node.parentId]) {
      // Root node or orphan — check if there are sibling roots
      // For simplicity, root nodes have no siblings
      return { siblings: [nodeId], index: 0, total: 1 };
    }
    const parent = session.tree[node.parentId];
    const siblings = parent.children.filter(
      (cid) => session.tree[cid] && session.tree[cid].role === node.role
    );
    const index = siblings.indexOf(nodeId);
    return { siblings, index: Math.max(0, index), total: siblings.length };
  }

  function getDeepestLeaf(session, nodeId) {
    let current = nodeId;
    while (true) {
      const node = session.tree[current];
      if (!node) return current;
      // Filter out image children — they're not on the conversation path
      const conversationChildren = node.children.filter(
        (cid) => session.tree[cid] && session.tree[cid].role !== "image"
      );
      if (conversationChildren.length === 0) return current;
      // Follow the last (most recent) conversation child
      current = conversationChildren[conversationChildren.length - 1];
    }
  }

  function switchToSibling(session, nodeId, direction) {
    const info = getSiblingInfo(session, nodeId);
    if (info.total <= 1) return session.activeLeafId;
    const newIndex = info.index + direction;
    if (newIndex < 0 || newIndex >= info.total) return session.activeLeafId;
    const newSiblingId = info.siblings[newIndex];
    session.activeLeafId = getDeepestLeaf(session, newSiblingId);
    return session.activeLeafId;
  }

  function migrateSessionToV2(session) {
    if (session.version === 2) return session;
    const tree = {};
    let prevId = null;
    const messages = session.messages || [];
    let rootId = null;
    let lastId = null;

    for (const msg of messages) {
      const id = generateNodeId();
      tree[id] = {
        id,
        role: msg.role,
        content: msg.content,
        parentId: prevId,
        children: [],
        createdAt: session.createdAt || Date.now(),
      };
      if (prevId && tree[prevId]) {
        tree[prevId].children.push(id);
      }
      if (!rootId) rootId = id;
      lastId = id;
      prevId = id;
    }

    session.version = 2;
    session.tree = tree;
    session.rootId = rootId;
    session.activeLeafId = lastId;
    delete session.messages;
    return session;
  }

  function sessionHasMessages(session) {
    if (!session) return false;
    if (session.version === 2) {
      return session.rootId && session.tree && Object.keys(session.tree).length > 0;
    }
    return session.messages && session.messages.length > 0;
  }

  function countDescendants(session, nodeId) {
    let count = 0;
    const stack = [...(session.tree[nodeId]?.children || [])];
    while (stack.length) {
      const cid = stack.pop();
      const child = session.tree[cid];
      if (child) {
        count++;
        stack.push(...child.children);
      }
    }
    return count;
  }

  function removeNodeAndDescendants(session, nodeId) {
    // Collect all descendant IDs
    const toRemove = [nodeId];
    const stack = [nodeId];
    while (stack.length) {
      const cid = stack.pop();
      const child = session.tree[cid];
      if (child) {
        for (const grandchild of child.children) {
          toRemove.push(grandchild);
          stack.push(grandchild);
        }
      }
    }
    // Remove from parent's children array
    const node = session.tree[nodeId];
    if (node && node.parentId && session.tree[node.parentId]) {
      const parent = session.tree[node.parentId];
      parent.children = parent.children.filter((cid) => cid !== nodeId);
    }
    // Delete all nodes
    for (const rid of toRemove) {
      delete session.tree[rid];
    }
    // Handle root deletion
    if (session.rootId === nodeId) {
      session.rootId = null;
    }
  }

  // ---- Settings (persisted in localStorage) ----

  let appSettings = {
    backendUrl: "",
    defaultModel: "default",
    defaultMode: "passthrough",
    theme: "dark",
    systemPrompt: "",
    temperature: null,
    maxTokens: null,
    topP: null,
    frequencyPenalty: null,
    presencePenalty: null,
    seed: null,
    stopSequences: "",
    contextLimit: null,
    sidebarOpen: true,
    imagePanelOpen: false,
    imgWidth: null,
    imgHeight: null,
    imgSteps: null,
    imgCfg: null,
    imgSeed: null,
    imgSampler: "",
    imgModel: "",
    imgPreset: "",
    imgNegative: "",
  };

  // ---- DOM References ----

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const dom = {
    app: $("#app"),
    sidebar: $("#sidebar"),
    sidebarOverlay: $("#sidebar-overlay"),
    toggleSidebarBtn: $("#toggle-sidebar-btn"),
    sidebarCloseBtn: $("#sidebar-close-btn"),
    newChatBtn: $("#new-chat-btn"),
    sessionList: $("#session-list"),
    modeBadge: $("#mode-badge"),
    settingsBtn: $("#settings-btn"),
    modelSelector: $("#model-selector"),
    messages: $("#messages"),
    messagesInner: $("#messages-inner"),
    welcome: $("#welcome"),
    chatInput: $("#chat-input"),
    sendBtn: $("#send-btn"),
    // Panels
    toggleReasoningBtn: $("#toggle-reasoning-btn"),
    toggleNarrativeBtn: $("#toggle-narrative-btn"),
    reasoningPanel: $("#reasoning-panel"),
    narrativePanel: $("#narrative-panel"),
    closeReasoningBtn: $("#close-reasoning-btn"),
    closeNarrativeBtn: $("#close-narrative-btn"),
    reasoningContent: $("#reasoning-content"),
    // Flow editor
    reasoningEditorView: $("#reasoning-editor-view"),
    reasoningLiveView: $("#reasoning-live-view"),
    flowSelect: $("#flow-select"),
    flowNewBtn: $("#flow-new-btn"),
    flowInfoName: $("#flow-info-name"),
    flowInfoDesc: $("#flow-info-desc"),
    flowBadgeDefault: $("#flow-badge-default"),
    flowBadgeBuiltin: $("#flow-badge-builtin"),
    flowCloneBtn: $("#flow-clone-btn"),
    flowSetDefaultBtn: $("#flow-set-default-btn"),
    flowExportBtn: $("#flow-export-btn"),
    flowDeleteBtn: $("#flow-delete-btn"),
    flowAddStepBtn: $("#flow-add-step-btn"),
    flowStepList: $("#flow-step-list"),
    flowStepEditor: $("#flow-step-editor"),
    flowStepsContainer: $("#flow-steps-container"),
    stepEditorBack: $("#step-editor-back"),
    stepNameInput: $("#step-name-input"),
    stepDeleteBtn: $("#step-delete-btn"),
    stepRoleSelect: $("#step-role-select"),
    stepSystemPrompt: $("#step-system-prompt"),
    stepSystemPromptRevert: $("#step-system-prompt-revert"),
    stepUserTemplate: $("#step-user-template"),
    stepUserTemplateRevert: $("#step-user-template-revert"),
    stepToolsGrid: $("#step-tools-grid"),
    stepOutputCap: $("#step-output-cap"),
    stepStreamToUser: $("#step-stream-to-user"),
    stepGateSimple: $("#step-gate-simple"),
    stepGateModerate: $("#step-gate-moderate"),
    stepGateComplex: $("#step-gate-complex"),
    stepSaveBtn: $("#step-save-btn"),
    flowImportBtn: $("#flow-import-btn"),
    flowImportFile: $("#flow-import-file"),
    liveFlowName: $("#live-flow-name"),
    liveFlowComplexity: $("#live-flow-complexity"),
    liveStepList: $("#live-step-list"),
    liveStats: $("#live-stats"),
    reasoningViewToggle: $("#reasoning-view-toggle"),
    reasoningPanelTitle: $("#reasoning-panel-title"),
    narrativeContent: $("#narrative-content"),
    // Settings modal
    settingsModal: $("#settings-modal"),
    settingsClose: $("#settings-close"),
    settingsCancel: $("#settings-cancel"),
    settingsSave: $("#settings-save"),
    settingBackendUrl: $("#setting-backend-url"),
    settingDefaultModel: $("#setting-default-model"),
    settingsTabs: $("#settings-tabs"),
    settingSystemPrompt: $("#setting-system-prompt"),
    settingTemperature: $("#setting-temperature"),
    settingTemperatureSlider: $("#setting-temperature-slider"),
    settingMaxTokens: $("#setting-max-tokens"),
    settingTopP: $("#setting-top-p"),
    settingTopPSlider: $("#setting-top-p-slider"),
    settingFreqPenalty: $("#setting-freq-penalty"),
    settingFreqPenaltySlider: $("#setting-freq-penalty-slider"),
    settingPresPenalty: $("#setting-pres-penalty"),
    settingPresPenaltySlider: $("#setting-pres-penalty-slider"),
    settingSeed: $("#setting-seed"),
    settingStop: $("#setting-stop"),
    settingContextLimit: $("#setting-context-limit"),
    // HuggingFace token
    settingHfToken: $("#setting-hf-token"),
    hfTokenStatus: $("#hf-token-status"),
    // Memory panel
    memoryStatTotal: $("#memory-stat-total"),
    memoryStats: $("#memory-stats"),
    memorySearchInput: $("#memory-search-input"),
    memorySearchBtn: $("#memory-search-btn"),
    memoryList: $("#memory-list"),
    memoryAddContent: $("#memory-add-content"),
    memoryAddType: $("#memory-add-type"),
    memoryAddBtn: $("#memory-add-btn"),
    memoryExportBtn: $("#memory-export-btn"),
    memoryCompactBtn: $("#memory-compact-btn"),
    // MCP panel
    mcpServerList: $("#mcp-server-list"),
    mcpConnectName: $("#mcp-connect-name"),
    mcpConnectType: $("#mcp-connect-type"),
    mcpConnectTarget: $("#mcp-connect-target"),
    mcpConnectArgs: $("#mcp-connect-args"),
    mcpConnectBtn: $("#mcp-connect-btn"),
    mcpToolList: $("#mcp-tool-list"),
    modeSwitcher: $("#mode-switcher"),
    modePopup: $("#mode-popup"),
    // Model Manager modal
    manageModelsBtn: $("#manage-models-btn"),
    modelManagerModal: $("#model-manager-modal"),
    modelManagerClose: $("#model-manager-close"),
    mmPullInput: $("#mm-pull-input"),
    mmPullBtn: $("#mm-pull-btn"),
    mmProgressArea: $("#mm-progress-area"),
    mmProgressModel: $("#mm-progress-model"),
    mmProgressFill: $("#mm-progress-fill"),
    mmProgressStatus: $("#mm-progress-status"),
    mmCancelBtn: $("#mm-cancel-btn"),
    mmModelList: $("#mm-model-list"),
    mmChips: $("#mm-chips"),
    mmBackendSelect: $("#mm-backend-select"),
    mmBrowseLink: $("#mm-browse-link"),
    mmGgufPicker: $("#mm-gguf-picker"),
    mmGgufList: $("#mm-gguf-list"),
    // Provider Manager modal
    settingsOpenProviders: $("#settings-open-providers"),
    providerModal: $("#provider-modal"),
    providerModalClose: $("#provider-modal-close"),
    provName: $("#prov-name"),
    provUrl: $("#prov-url"),
    provKey: $("#prov-key"),
    provTestBtn: $("#prov-test-btn"),
    provAddBtn: $("#prov-add-btn"),
    provTestResult: $("#prov-test-result"),
    provList: $("#prov-list"),
  };

  // ---- Initialization ----

  function init() {
    loadSettings();
    loadSessions();
    applyTheme(appSettings.theme);
    selectedModel = appSettings.defaultModel;
    currentMode = appSettings.defaultMode;
    updateModeBadge();
    renderSessionList();

    // Restore last active session
    const lastActive = localStorage.getItem(STORAGE_ACTIVE);
    if (lastActive && sessions[lastActive]) {
      switchSession(lastActive);
    }

    fetchModels();
    fetchCapabilities();
    startConnectionMonitor();
    bindEvents();
    autoGrowTextarea(dom.chatInput);
    loadFlows();

    // Listen for narrative state updates from narrative/index.js poller
    // (avoids duplicate API calls — only one poller hits the backend)
    document.addEventListener('augmentum:narrative-state', (e) => {
      const { data, sessionId } = e.detail;
      const hasState = !!(data && data.state);
      const isExternal = hasState && currentMode !== 'narrative';
      updateNarrativeIndicator(hasState, isExternal);
      renderNarrativeState(data, sessionId);
    });
  }

  // ---- Persistence ----

  function loadSessions() {
    try {
      const raw = localStorage.getItem(STORAGE_SESSIONS);
      sessions = raw ? JSON.parse(raw) : {};
    } catch {
      sessions = {};
    }
    // Migrate v1 sessions (flat messages[]) to v2 (tree-based)
    let migrated = false;
    for (const id of Object.keys(sessions)) {
      if (sessions[id].version !== 2) {
        sessions[id] = migrateSessionToV2(sessions[id]);
        migrated = true;
      }
    }
    if (migrated) saveSessions();
  }

  function saveSessions() {
    localStorage.setItem(STORAGE_SESSIONS, JSON.stringify(sessions));
  }

  function loadSettings() {
    try {
      const raw = localStorage.getItem(STORAGE_SETTINGS);
      if (raw) {
        appSettings = { ...appSettings, ...JSON.parse(raw) };
      }
    } catch {
      /* ignore */
    }
  }

  function saveSettings() {
    localStorage.setItem(STORAGE_SETTINGS, JSON.stringify(appSettings));
  }

  // ---- Theme ----

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    appSettings.theme = theme;
    // Update theme toggle buttons
    $$(".theme-option").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.theme === theme);
    });
    // Switch highlight.js theme
    const hljsLink = document.getElementById("hljs-theme");
    if (hljsLink) {
      const hljsTheme = theme === "light" ? "github" : "github-dark";
      hljsLink.href = `lib/highlight.js/${hljsTheme}.min.css`;
    }
  }

  // ---- Connection Status ----

  function setConnectionStatus(status) {
    const el = document.getElementById("connection-status");
    if (!el) return;
    el.className = `connection-status ${status}`;
    const labels = { connected: "Connected", disconnected: "Disconnected", checking: "Checking..." };
    el.title = labels[status] || status;
  }

  async function checkConnection() {
    setConnectionStatus("checking");
    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(`${base}/`, { method: "GET", signal: AbortSignal.timeout(5000) });
      setConnectionStatus(resp.ok ? "connected" : "disconnected");
    } catch {
      setConnectionStatus("disconnected");
    }
  }

  // Periodic connection check (every 30s)
  let _connectionInterval = null;
  function startConnectionMonitor() {
    checkConnection();
    if (_connectionInterval) clearInterval(_connectionInterval);
    _connectionInterval = setInterval(() => { checkConnection(); fetchCapabilities(); }, 30000);
  }

  // ---- Models ----

  async function fetchModels() {
    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(`${base}/api/tags`);
      if (!resp.ok) throw new Error("fetch failed");
      const data = await resp.json();
      const models = data.models || [];
      populateModelSelector(models);
      setConnectionStatus("connected");
    } catch {
      // Show a default option
      dom.modelSelector.innerHTML = '<option value="default">default</option>';
      setConnectionStatus("disconnected");
    }
  }

  // Server capabilities — adapts UI to what's available
  let serverCapabilities = { image_enabled: false, memory_enabled: true, mcp_enabled: true, backends: [], has_backends: false };

  async function fetchCapabilities() {
    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(`${base}/api/capabilities`);
      if (!resp.ok) return;
      serverCapabilities = await resp.json();
    } catch { /* ignore — use defaults */ }

    // If image_enabled is still false, check for cloud image providers
    if (!serverCapabilities.image_enabled) {
      try {
        const cloudResp = await fetch('/api/image/cloud/providers');
        if (cloudResp.ok) {
          const providers = await cloudResp.json();
          if (providers.length > 0) {
            serverCapabilities.image_enabled = true;
          }
        }
      } catch { /* ignore */ }
    }

    applyCapabilities();
  }

  function applyCapabilities() {
    // Image generation panel toggle
    const imgBtn = $("#toggle-image-btn");
    if (imgBtn) {
      imgBtn.style.display = serverCapabilities.image_enabled ? "" : "none";
    }
    // If image panel is open but images disabled, close it
    if (!serverCapabilities.image_enabled) {
      const imgPanel = $("#image-panel");
      if (imgPanel && !imgPanel.classList.contains("hidden")) {
        imgPanel.classList.add("hidden");
      }
    }

    // Hide per-message image gen buttons when images are disabled
    document.body.classList.toggle("no-image-gen", !serverCapabilities.image_enabled);

    // No backends connected — update welcome screen with setup guidance
    const welcome = $("#welcome");
    if (welcome && !serverCapabilities.has_backends) {
      // Show a helpful banner above the existing welcome content
      let banner = welcome.querySelector(".no-backend-banner");
      if (!banner) {
        banner = document.createElement("div");
        banner.className = "no-backend-banner";
        welcome.insertBefore(banner, welcome.firstChild);
      }
      banner.innerHTML = `
        <div style="background: var(--surface); border: 1px solid var(--warning); border-radius: 10px; padding: 16px 20px; margin-bottom: 16px; max-width: 480px; text-align: left;">
          <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--warning)" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
            <strong style="color: var(--warning); font-size: 13px;">No model backends connected</strong>
          </div>
          <p style="color: var(--text-secondary); font-size: 13px; line-height: 1.6; margin: 0 0 10px;">
            Connect an external API or start a bundled backend:
          </p>
          <ul style="color: var(--text-secondary); font-size: 12px; line-height: 1.8; margin: 0; padding-left: 18px;">
            <li><strong>External API</strong> &mdash; Settings &rarr; Manage Providers (LM Studio, OpenAI, OpenRouter, etc.)</li>
            <li><strong>Bundled Ollama</strong> &mdash; <code style="font-size: 11px;">docker compose --profile ollama up</code></li>
            <li><strong>Bundled llama.cpp</strong> &mdash; <code style="font-size: 11px;">docker compose --profile llamacpp up</code></li>
          </ul>
        </div>
      `;
    } else if (welcome) {
      const banner = welcome.querySelector(".no-backend-banner");
      if (banner) banner.remove();
    }
  }

  function populateModelSelector(models) {
    const sel = dom.modelSelector;
    const settingSel = dom.settingDefaultModel;
    sel.innerHTML = "";
    settingSel.innerHTML = '<option value="default">default</option>';

    // Filter out prefixed (a/, n/, p/) for main selector — show only base models
    const baseModels = models.filter(
      (m) => !m.name.startsWith("a/") && !m.name.startsWith("n/") && !m.name.startsWith("p/")
    );

    if (baseModels.length === 0) {
      sel.innerHTML = '<option value="default">default</option>';
      return;
    }

    baseModels.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.name;
      const backend = m.details && m.details.augmentum_backend;
      opt.textContent = backend && backend !== "ollama"
        ? `${m.name} [${backend}]`
        : m.name;
      sel.appendChild(opt);

      const opt2 = opt.cloneNode(true);
      settingSel.appendChild(opt2);
    });

    // Restore selected model
    if (selectedModel && sel.querySelector(`option[value="${CSS.escape(selectedModel)}"]`)) {
      sel.value = selectedModel;
    } else {
      selectedModel = sel.value;
    }

    // Populate prompt condense model selector (image tab)
    var condenseSel = $("#img-condense-model");
    if (condenseSel) {
      condenseSel.innerHTML = '<option value="">Default (backend default)</option>';
      baseModels.forEach(function (m) {
        var opt = document.createElement("option");
        opt.value = m.name;
        var backend = m.details && m.details.augmentum_backend;
        opt.textContent = backend && backend !== "ollama"
          ? m.name + " [" + backend + "]"
          : m.name;
        condenseSel.appendChild(opt);
      });
      if (appSettings.imgCondenseModel) {
        condenseSel.value = appSettings.imgCondenseModel;
      }
    }
  }

  // ---- Session Management ----

  function createSession() {
    const id = "s_" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    sessions[id] = {
      id,
      title: "New Chat",
      version: 2,
      tree: {},
      rootId: null,
      activeLeafId: null,
      createdAt: Date.now(),
    };
    saveSessions();
    renderSessionList();
    switchSession(id);
    return id;
  }

  function deleteSession(id) {
    delete sessions[id];
    saveSessions();
    if (activeSessionId === id) {
      activeSessionId = null;
      localStorage.removeItem(STORAGE_ACTIVE);
      renderMessages();
    }
    renderSessionList();
  }

  function switchSession(id) {
    if (!sessions[id]) return;
    activeSessionId = id;
    localStorage.setItem(STORAGE_ACTIVE, id);
    renderSessionList();
    renderMessages();
    dom.chatInput.focus();
  }

  function getActiveSession() {
    return activeSessionId ? sessions[activeSessionId] : null;
  }

  function updateSessionTitle(id) {
    const session = sessions[id];
    if (!session || !sessionHasMessages(session)) return;
    // Use the first user message as title (truncated)
    // Walk from root to find first user node
    let firstUserContent = null;
    if (session.version === 2 && session.rootId) {
      const root = session.tree[session.rootId];
      if (root && root.role === "user") {
        firstUserContent = root.content;
      }
    }
    if (firstUserContent) {
      session.title = firstUserContent.slice(0, 60) + (firstUserContent.length > 60 ? "..." : "");
      saveSessions();
      renderSessionList();
    }
  }

  // ---- Rendering ----

  function renderSessionList() {
    const list = dom.sessionList;
    list.innerHTML = "";

    // Sort sessions by creation time, newest first
    const sorted = Object.values(sessions).sort((a, b) => b.createdAt - a.createdAt);

    sorted.forEach((s) => {
      const item = document.createElement("div");
      item.className = "session-item" + (s.id === activeSessionId ? " active" : "");
      item.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        <span class="session-title">${escapeHtml(s.title)}</span>
        <button class="session-delete" data-id="${s.id}" title="Delete">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      `;
      item.addEventListener("click", (e) => {
        if (e.target.closest(".session-delete")) return;
        switchSession(s.id);
      });
      item.querySelector(".session-delete").addEventListener("click", (e) => {
        e.stopPropagation();
        deleteSession(s.id);
      });
      list.appendChild(item);
    });
  }

  function renderMessages() {
    const session = getActiveSession();
    dom.messagesInner.innerHTML = "";

    if (!session || !sessionHasMessages(session)) {
      dom.messagesInner.appendChild(createWelcomeEl());
      return;
    }

    const path = getActivePath(session);
    path.forEach((node) => {
      dom.messagesInner.appendChild(createMessageEl(node, session));
      // Render inline images that are children of this node
      if (node.children && node.children.length > 0) {
        node.children.forEach((childId) => {
          const child = session.tree[childId];
          if (child && child.role === "image") {
            dom.messagesInner.appendChild(createInlineImageEl(child));
          }
        });
      }
    });
    updateRegenerateButtons();
    highlightCode(dom.messagesInner);
    restoreReasoningPanel(path);
    scrollToBottom(true);
  }

  function restoreReasoningPanel(path) {
    // Find the last assistant node with reasoning data
    let reasoning = null;
    for (let i = path.length - 1; i >= 0; i--) {
      if (path[i].role === "assistant" && path[i].reasoning) {
        reasoning = path[i].reasoning;
        break;
      }
    }

    if (!reasoning || !reasoning.phases || reasoning.phases.length === 0) {
      showDefaultReasoningPhases();
      return;
    }

    // Rebuild phases with content from the persisted phaseContent map
    const phases = reasoning.phases.map((p) => ({
      name: p.name,
      status: p.status,
      output: (reasoning.phaseContent && reasoning.phaseContent[p.name]) || p.output || "",
    }));

    renderReasoningPhases(phases, reasoning.complexity);
  }

  function inspectMessageReasoning(nodeId) {
    const session = sessions[activeSessionId];
    if (!session || !session.tree) return;
    const node = session.tree[nodeId];
    if (!node || !node.reasoning) return;

    const reasoning = node.reasoning;
    const phases = reasoning.phases.map((p) => ({
      name: p.name,
      status: p.status,
      output: (reasoning.phaseContent && reasoning.phaseContent[p.name]) || p.output || "",
    }));

    // Show the reasoning panel if hidden
    dom.reasoningPanel.classList.remove("hidden");

    // Render phases into the inspector
    renderReasoningPhases(phases, reasoning.complexity);

    // Highlight which message is being inspected
    document.querySelectorAll(".message.reasoning-inspected").forEach((el) => {
      el.classList.remove("reasoning-inspected");
    });
    const msgEl = document.querySelector(`.message[data-node-id="${nodeId}"]`);
    if (msgEl) msgEl.classList.add("reasoning-inspected");
  }

  function createWelcomeEl() {
    const div = document.createElement("div");
    div.className = "welcome";
    div.id = "welcome";
    div.innerHTML = `
      <h2>Augmentum</h2>
      <p>LLM API proxy with narrative intelligence. Start a conversation below.</p>
      <div class="shortcuts">
        <button class="shortcut-chip" data-prompt="Tell me a story about a brave knight.">Tell a story</button>
        <button class="shortcut-chip" data-prompt="What can you help me with?">What can you do?</button>
        <button class="shortcut-chip" data-prompt="Explain how Augmentum processes messages.">How does this work?</button>
      </div>
    `;
    div.querySelectorAll(".shortcut-chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        dom.chatInput.value = btn.dataset.prompt;
        handleSend();
      });
    });
    return div;
  }

  // SVG icon set for message action buttons
  const icons = {
    copy: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>',
    check: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    image: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>',
    regen: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>',
    trash: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>',
    edit: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.83 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"/></svg>',
    download: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
    chevronLeft: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"/></svg>',
    chevronRight: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>',
  };

  // Build a static thinking block from persisted reasoning data (shown on re-render / reload)
  function buildStoredThinkingHtml(reasoning) {
    if (!reasoning || !reasoning.phases || reasoning.phases.length === 0) return "";

    const badge = reasoning.complexity
      ? `<span class="thinking-complexity thinking-complexity-${escapeHtml(reasoning.complexity)}">${escapeHtml(reasoning.complexity)}</span>`
      : "";

    const completedCount = reasoning.phases.filter((p) => p.status === "complete").length;
    const totalCount = reasoning.phases.length;
    const allComplete = totalCount > 0 && completedCount === totalCount;
    const headerLabel = allComplete
      ? `Analyzed in ${totalCount} phases`
      : `Analyzing\u2026 (${completedCount}/${totalCount})`;

    const phaseItems = reasoning.phases.map((p) => {
      const statusIcon = p.status === "complete" ? "&#10003;" : p.status === "running" ? "&#9679;" : "&#9675;";
      const phaseText = (reasoning.phaseContent && reasoning.phaseContent[p.name]) || p.output || "";
      const hasContent = phaseText.length > 0;
      const toggleBtn = hasContent
        ? `<button class="phase-expand-btn" data-action="toggle-phase-content"><span class="phase-expand-icon">\u25B6</span></button>`
        : "";
      const contentHtml = hasContent
        ? `<div class="thinking-phase-content collapsed" data-phase="${escapeHtml(p.name)}">${renderMarkdown(phaseText)}</div>`
        : "";
      const displayName = PHASE_DISPLAY_NAMES[p.name] || p.name;
      return `<div class="thinking-phase ${p.status}" data-phase="${escapeHtml(p.name)}"><span class="thinking-phase-icon">${statusIcon}</span><span class="thinking-phase-name">${escapeHtml(displayName)}</span>${toggleBtn}</div>${contentHtml}`;
    }).join("");

    return `<div class="thinking-block">
      <div class="thinking-header" data-toggle-parent="open">
        <svg class="thinking-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><polyline points="6 9 12 15 18 9"/></svg>
        <span class="thinking-label">${headerLabel}</span>
        ${badge}
      </div>
      <div class="thinking-body">${phaseItems}</div>
    </div>`;
  }

  // Build static tool call cards from persisted reasoning data
  function buildStoredToolCallsHtml(toolCalls) {
    if (!toolCalls || toolCalls.length === 0) return "";

    const passCount = toolCalls.filter((tc) => tc.success).length;
    const failCount = toolCalls.length - passCount;
    const summaryParts = [];
    if (passCount) summaryParts.push(`${passCount} passed`);
    if (failCount) summaryParts.push(`${failCount} failed`);
    const toolNames = [...new Set(toolCalls.map((tc) => tc.tool))].join(", ");

    const cards = toolCalls.map((tc) => {
      const successClass = tc.success ? "success" : "error";
      const inputStr = tc.input ? JSON.stringify(tc.input, null, 2) : "";
      const outputStr = tc.output || "No output";
      const inputSection = inputStr ? buildExpandableSection("Input", inputStr) : "";
      const outputSection = buildExpandableSection("Output", outputStr);

      return `<div class="tool-call-card">
        <div class="tool-call-header" data-toggle-parent="open">
          <svg class="tool-call-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><polyline points="6 9 12 15 18 9"/></svg>
          <span class="tool-call-icon ${successClass}">${tc.success ? "&#10003;" : "&#10007;"}</span>
          <span class="tool-call-name">${escapeHtml(tc.tool)}</span>
          <span class="tool-call-phase">${escapeHtml(tc.phase)}</span>
        </div>
        <div class="tool-call-body">
          ${inputSection}
          ${outputSection}
        </div>
      </div>`;
    }).join("");

    return `<div class="tool-calls-container">
      <div class="tool-calls-summary" data-toggle-parent="open">
        <svg class="thinking-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><polyline points="6 9 12 15 18 9"/></svg>
        <span class="tool-calls-summary-label">${toolCalls.length} tool call${toolCalls.length > 1 ? "s" : ""}</span>
        <span class="tool-calls-summary-detail">${toolNames} &mdash; ${summaryParts.join(", ")}</span>
      </div>
      <div class="tool-calls-list">${cards}</div>
    </div>`;
  }

  function createMessageEl(node, session) {
    const role = node.role;
    const content = node.content;
    const msg = document.createElement("div");
    msg.className = "message " + role;
    msg.dataset.nodeId = node.id;

    const avatarLabel = role === "user" ? "U" : "A";

    // Build branch navigation if this node has siblings
    let branchNavHtml = "";
    if (session) {
      const info = getSiblingInfo(session, node.id);
      if (info.total > 1) {
        const prevDisabled = info.index <= 0 ? " disabled" : "";
        const nextDisabled = info.index >= info.total - 1 ? " disabled" : "";
        branchNavHtml = `<div class="branch-nav">
          <button class="branch-nav-btn branch-prev" data-action="branch-prev" data-node-id="${node.id}"${prevDisabled}>${icons.chevronLeft}</button>
          <span class="branch-counter">${info.index + 1} / ${info.total}</span>
          <button class="branch-nav-btn branch-next" data-action="branch-next" data-node-id="${node.id}"${nextDisabled}>${icons.chevronRight}</button>
        </div>`;
      }
    }

    // Build action buttons
    let actionsHtml = "";
    if (role === "assistant") {
      const reasoningBtn = node.reasoning && node.reasoning.phases && node.reasoning.phases.length > 0
        ? `<button class="message-action-btn inspect-reasoning-btn" data-action="inspect-reasoning" data-node-id="${node.id}" title="Inspect reasoning phases"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg><span>Reasoning</span></button>`
        : "";
      actionsHtml = `<div class="message-actions">
          <button class="message-action-btn copy-msg-btn" title="Copy response">${icons.copy}<span>Copy</span></button>
          ${reasoningBtn}
          <button class="message-action-btn generate-img-btn" data-action="generate-image" data-node-id="${node.id}" title="Generate image from this message">${icons.image}<span>Image</span></button>
          <button class="message-action-btn regenerate-btn" data-action="regenerate-message" data-node-id="${node.id}" title="Regenerate response">${icons.regen}<span>Regen</span></button>
          <button class="message-action-btn delete-msg-btn" data-action="delete-message" data-node-id="${node.id}" title="Delete message">${icons.trash}<span>Delete</span></button>
        </div>`;
    } else {
      actionsHtml = `<div class="message-actions">
          <button class="message-action-btn generate-img-btn" data-action="generate-image" data-node-id="${node.id}" title="Generate image from this message">${icons.image}<span>Image</span></button>
          <button class="message-action-btn edit-msg-btn" data-action="edit-message" data-node-id="${node.id}" title="Edit message">${icons.edit}<span>Edit</span></button>
          <button class="message-action-btn delete-msg-btn" data-action="delete-message" data-node-id="${node.id}" title="Delete message">${icons.trash}<span>Delete</span></button>
        </div>`;
    }

    // Build message content — assistant messages with reasoning get thinking block + tool cards
    let contentInnerHtml;
    if (role === "assistant" && node.reasoning) {
      const thinkingHtml = buildStoredThinkingHtml(node.reasoning);
      const toolCallsHtml = buildStoredToolCallsHtml(node.reasoning.toolCalls);
      contentInnerHtml = `${thinkingHtml}${toolCallsHtml}<div class="response-body">${renderMarkdown(content)}</div>`;
    } else if (role === "user") {
      contentInnerHtml = escapeHtml(content).replace(/\n/g, "<br>");
    } else {
      contentInnerHtml = renderMarkdown(content);
    }

    msg.innerHTML = `
      <div class="message-avatar">${avatarLabel}</div>
      <div class="message-body">
        ${branchNavHtml}
        <div class="message-content">${contentInnerHtml}</div>
      </div>
      ${actionsHtml}
    `;

    // Wire up copy button for assistant messages
    if (role === "assistant") {
      const copyBtn = msg.querySelector(".copy-msg-btn");
      if (copyBtn) {
        copyBtn.addEventListener("click", () => {
          navigator.clipboard.writeText(content).then(() => {
            copyBtn.innerHTML = icons.check + "<span>Copied</span>";
            copyBtn.classList.add("copied");
            setTimeout(() => {
              copyBtn.innerHTML = icons.copy + "<span>Copy</span>";
              copyBtn.classList.remove("copied");
            }, 1500);
          });
        });
      }
    }
    return msg;
  }

  // Track tool calls accumulated during current stream
  let streamToolCalls = [];
  let streamPhases = [];
  let streamComplexity = "";
  let streamThinkingOpen = false;
  let streamPhaseContent = {}; // Maps phase name → accumulated text
  let streamModelThinking = ""; // Accumulated model thinking text

  function createStreamingMessageEl() {
    const msg = document.createElement("div");
    msg.className = "message assistant";
    msg.id = "streaming-message";
    msg.dataset.nodeId = "pending";
    msg.innerHTML = `
      <div class="message-avatar">A</div>
      <div class="message-body">
        <div class="message-content">
          <div class="typing-indicator">
            <span></span><span></span><span></span>
          </div>
        </div>
      </div>
    `;
    // Reset stream metadata
    streamToolCalls = [];
    streamPhases = [];
    streamComplexity = "";
    streamThinkingOpen = false;
    streamPhaseContent = {};
    streamModelThinking = "";
    return msg;
  }

  function updateStreamThinking(phases, complexity) {
    const el = document.getElementById("streaming-message");
    if (!el) return;
    const content = el.querySelector(".message-body .message-content") || el.querySelector(".message-content");

    // Remove typing indicator if still present
    const indicator = content.querySelector(".typing-indicator");
    if (indicator) indicator.remove();

    // Store phase info
    streamPhases = phases || streamPhases;
    streamComplexity = complexity || streamComplexity;

    // Clear content for phases that are (re-)running (e.g. backtracking)
    for (const p of streamPhases) {
      if (p.status === "running") {
        streamPhaseContent[p.name] = "";
      }
    }

    // Find or create thinking block
    let thinkingBlock = content.querySelector(".thinking-block");
    if (!thinkingBlock) {
      thinkingBlock = document.createElement("div");
      thinkingBlock.className = "thinking-block open";
      streamThinkingOpen = true;
      content.insertBefore(thinkingBlock, content.firstChild);
    }

    // Count completed phases for progress indicator
    const completedCount = streamPhases.filter((p) => p.status === "complete").length;
    const totalCount = streamPhases.length;

    // Build compact phase progress — only show phase names as a horizontal pipeline
    const phaseItems = streamPhases.map((p) => {
      const statusIcon =
        p.status === "complete" ? "&#10003;" : p.status === "running" ? "&#9679;" : "&#9675;";
      const phaseText = streamPhaseContent[p.name] || "";
      const hasContent = phaseText.length > 0;
      const toggleBtn = hasContent
        ? `<button class="phase-expand-btn" data-action="toggle-phase-content"><span class="phase-expand-icon">\u25B6</span></button>`
        : "";
      // Phase content starts collapsed — user expands if curious
      const contentHtml = `<div class="thinking-phase-content collapsed" data-phase="${escapeHtml(p.name)}">${hasContent ? renderMarkdown(phaseText) : ""}</div>`;
      const displayName = PHASE_DISPLAY_NAMES[p.name] || p.name;
      return `<div class="thinking-phase ${p.status}" data-phase="${escapeHtml(p.name)}"><span class="thinking-phase-icon">${statusIcon}</span><span class="thinking-phase-name">${escapeHtml(displayName)}</span>${toggleBtn}</div>${contentHtml}`;
    }).join("");

    const runningPhase = streamPhases.find((p) => p.status === "running");
    const allComplete = totalCount > 0 && completedCount === totalCount;
    const headerLabel = allComplete
      ? "Analyzed in " + totalCount + " phases"
      : runningPhase
        ? `Analyzing\u2026 ${escapeHtml(PHASE_DISPLAY_NAMES[runningPhase.name] || runningPhase.name)} (${completedCount}/${totalCount})`
        : "Analyzing\u2026";

    const badge = streamComplexity
      ? `<span class="thinking-complexity thinking-complexity-${streamComplexity}">${escapeHtml(streamComplexity)}</span>`
      : "";

    thinkingBlock.innerHTML = `
      <div class="thinking-header" data-toggle-parent="open">
        <svg class="thinking-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><polyline points="6 9 12 15 18 9"/></svg>
        <span class="thinking-label">${headerLabel}</span>
        ${badge}
      </div>
      <div class="thinking-body">${phaseItems}</div>
    `;
    scrollToBottom();
  }

  function appendPhaseContent(phaseName, delta) {
    // Accumulate in state
    if (!streamPhaseContent[phaseName]) streamPhaseContent[phaseName] = "";
    streamPhaseContent[phaseName] += delta;

    const el = document.getElementById("streaming-message");
    if (!el) return;

    // Remove typing indicator if still present
    const msgContent = el.querySelector(".message-body .message-content") || el.querySelector(".message-content");
    const indicator = msgContent ? msgContent.querySelector(".typing-indicator") : null;
    if (indicator) indicator.remove();

    // Find the content div for this phase (created by updateStreamThinking)
    const contentDiv = el.querySelector(`.thinking-phase-content[data-phase="${phaseName}"]`);
    if (contentDiv) {
      // Render as markdown for better readability
      contentDiv.innerHTML = renderMarkdown(streamPhaseContent[phaseName]);
      // Auto-scroll the content area to the bottom
      contentDiv.scrollTop = contentDiv.scrollHeight;
    }

    // Auto-scroll the thinking body
    const thinkingBody = el.querySelector(".thinking-body");
    if (thinkingBody) {
      thinkingBody.scrollTop = thinkingBody.scrollHeight;
    }
    scrollToBottom();
  }

  function addStreamToolCall(toolCall) {
    streamToolCalls.push(toolCall);
    const el = document.getElementById("streaming-message");
    if (!el) return;
    const content = el.querySelector(".message-body .message-content") || el.querySelector(".message-content");

    // Find or create tool calls container
    let toolContainer = content.querySelector(".tool-calls-container");
    if (!toolContainer) {
      toolContainer = document.createElement("div");
      toolContainer.className = "tool-calls-container open";
      toolContainer.innerHTML = `
        <div class="tool-calls-summary" data-toggle-parent="open">
          <svg class="thinking-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><polyline points="6 9 12 15 18 9"/></svg>
          <span class="tool-calls-summary-label"></span>
          <span class="tool-calls-summary-detail"></span>
        </div>
        <div class="tool-calls-list"></div>
      `;
      // Insert after thinking block if it exists, otherwise at start
      const thinkingBlock = content.querySelector(".thinking-block");
      if (thinkingBlock) {
        thinkingBlock.after(toolContainer);
      } else {
        content.insertBefore(toolContainer, content.firstChild);
      }
    }

    // Update summary bar
    const passCount = streamToolCalls.filter((tc) => tc.success).length;
    const failCount = streamToolCalls.length - passCount;
    const summaryParts = [];
    if (passCount) summaryParts.push(`${passCount} passed`);
    if (failCount) summaryParts.push(`${failCount} failed`);
    const toolNames = [...new Set(streamToolCalls.map((tc) => tc.tool))].join(", ");
    const labelEl = toolContainer.querySelector(".tool-calls-summary-label");
    const detailEl = toolContainer.querySelector(".tool-calls-summary-detail");
    if (labelEl) labelEl.textContent = `${streamToolCalls.length} tool call${streamToolCalls.length > 1 ? "s" : ""}`;
    if (detailEl) detailEl.innerHTML = `${escapeHtml(toolNames)} &mdash; ${summaryParts.join(", ")}`;

    // Add tool call card to the list
    const list = toolContainer.querySelector(".tool-calls-list");
    const card = document.createElement("div");
    card.className = "tool-call-card";
    const successClass = toolCall.success ? "success" : "error";
    const inputStr = toolCall.input ? JSON.stringify(toolCall.input, null, 2) : "";
    const outputStr = toolCall.output || "No output";

    // Build expandable sections for input and output
    const inputSection = inputStr ? buildExpandableSection("Input", inputStr) : "";
    const outputSection = buildExpandableSection("Output", outputStr);

    card.innerHTML = `
      <div class="tool-call-header" data-toggle-parent="open">
        <svg class="tool-call-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><polyline points="6 9 12 15 18 9"/></svg>
        <span class="tool-call-icon ${successClass}">${toolCall.success ? "&#10003;" : "&#10007;"}</span>
        <span class="tool-call-name">${escapeHtml(toolCall.tool)}</span>
        <span class="tool-call-phase">${escapeHtml(toolCall.phase)}</span>
      </div>
      <div class="tool-call-body">
        ${inputSection}
        ${outputSection}
      </div>
    `;
    list.appendChild(card);
    scrollToBottom();
  }

  function appendModelThinking(delta) {
    const el = document.getElementById("streaming-message");
    if (!el) return;
    const content = el.querySelector(".message-body .message-content") || el.querySelector(".message-content");

    // Remove typing indicator if present
    const indicator = content.querySelector(".typing-indicator");
    if (indicator) indicator.remove();

    streamModelThinking += delta;

    // Find or create model thinking block
    let block = content.querySelector(".model-thinking-block");
    if (!block) {
      block = document.createElement("div");
      block.className = "model-thinking-block open";
      block.innerHTML = `
        <div class="thinking-header" data-toggle-parent="open">
          <svg class="thinking-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><polyline points="6 9 12 15 18 9"/></svg>
          <span class="thinking-label">Thinking\u2026</span>
          <span class="thinking-complexity thinking-complexity-model">model</span>
        </div>
        <div class="thinking-body">
          <pre class="model-thinking-content"></pre>
        </div>
      `;
      // Insert before UARF thinking block if present, otherwise at the top
      const uarfBlock = content.querySelector(".thinking-block");
      if (uarfBlock) {
        content.insertBefore(block, uarfBlock);
      } else {
        content.insertBefore(block, content.firstChild);
      }
    }

    const pre = block.querySelector(".model-thinking-content");
    if (pre) {
      pre.textContent = streamModelThinking;
    }
    scrollToBottom();
  }

  function appendToStreamingMessage(text) {
    const el = document.getElementById("streaming-message");
    if (!el) return;
    const content = el.querySelector(".message-body .message-content") || el.querySelector(".message-content");

    // Remove typing indicator if present
    const indicator = content.querySelector(".typing-indicator");
    if (indicator) {
      indicator.remove();
    }

    // Auto-collapse thinking blocks when actual content starts streaming
    if (text.trim()) {
      if (streamThinkingOpen) {
        const thinkingBlock = content.querySelector(".thinking-block");
        if (thinkingBlock) {
          thinkingBlock.classList.remove("open");
          streamThinkingOpen = false;
        }
      }
      const modelBlock = content.querySelector(".model-thinking-block.open");
      if (modelBlock) {
        modelBlock.classList.remove("open");
        const label = modelBlock.querySelector(".thinking-label");
        if (label) label.textContent = "Thought for a moment";
      }
      // Collapse tool calls when response content starts
      const toolContainer = content.querySelector(".tool-calls-container.open");
      if (toolContainer) {
        toolContainer.classList.remove("open");
      }
    }

    // We accumulate raw text and re-render markdown each time
    // Store raw text in a data attribute
    const current = el.dataset.rawContent || "";
    const updated = current + text;
    el.dataset.rawContent = updated;

    // Find or create response body (after thinking + tool blocks)
    let responseBody = content.querySelector(".response-body");
    if (!responseBody) {
      responseBody = document.createElement("div");
      responseBody.className = "response-body";
      content.appendChild(responseBody);
    }
    responseBody.innerHTML = renderMarkdown(updated);
    highlightCode(responseBody);
    scrollToBottom();
  }

  function finalizeStreamingMessage(session) {
    const el = document.getElementById("streaming-message");
    if (!el) return;
    const rawContent = el.dataset.rawContent || "";
    el.removeAttribute("id");
    el.removeAttribute("data-raw-content");

    // Final render of response body
    const responseBody = el.querySelector(".response-body");
    if (responseBody) {
      responseBody.innerHTML = renderMarkdown(rawContent);
      highlightCode(responseBody);
    }

    // Ensure thinking blocks are collapsed
    const thinkingBlock = el.querySelector(".thinking-block");
    if (thinkingBlock) {
      thinkingBlock.classList.remove("open");
    }
    const modelThinkingBlock = el.querySelector(".model-thinking-block");
    if (modelThinkingBlock) {
      modelThinkingBlock.classList.remove("open");
      const label = modelThinkingBlock.querySelector(".thinking-label");
      if (label) label.textContent = "Thought for a moment";
    }

    // Add message action buttons (copy + regenerate)
    // nodeId will be set to real value by sendMessage() after this returns
    const actionsDiv = document.createElement("div");
    actionsDiv.className = "message-actions";
    actionsDiv.innerHTML = `
      <button class="message-action-btn copy-msg-btn" title="Copy response">${icons.copy}<span>Copy</span></button>
      <button class="message-action-btn generate-img-btn" data-action="generate-image" data-node-id="pending" title="Generate image from this message">${icons.image}<span>Image</span></button>
      <button class="message-action-btn regenerate-btn" data-action="regenerate-message" data-node-id="pending" title="Regenerate response">${icons.regen}<span>Regen</span></button>
      <button class="message-action-btn delete-msg-btn" data-action="delete-message" data-node-id="pending" title="Delete message">${icons.trash}<span>Delete</span></button>
    `;
    el.appendChild(actionsDiv);

    const copyBtn = actionsDiv.querySelector(".copy-msg-btn");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(rawContent).then(() => {
          copyBtn.innerHTML = icons.check + "<span>Copied</span>";
          copyBtn.classList.add("copied");
          setTimeout(() => {
            copyBtn.innerHTML = icons.copy + "<span>Copy</span>";
            copyBtn.classList.remove("copied");
          }, 1500);
        });
      });
    }

    // Update regenerate buttons (only last assistant message should be active)
    updateRegenerateButtons();

    return rawContent;
  }

  let userScrolledUp = false;

  dom.messages.addEventListener("scroll", () => {
    const threshold = 80;
    const atBottom = dom.messages.scrollHeight - dom.messages.scrollTop - dom.messages.clientHeight < threshold;
    userScrolledUp = !atBottom;
  });

  function scrollToBottom(force) {
    if (!force && userScrolledUp) return;
    dom.messages.scrollTop = dom.messages.scrollHeight;
  }

  // ---- Mode Badge ----

  function updateModeBadge() {
    const badge = dom.modeBadge;
    badge.textContent = currentMode.toUpperCase();
    badge.className = "mode-badge " + currentMode;
  }

  function updateModePopupActiveState() {
    dom.modePopup.querySelectorAll(".mode-option").forEach((opt) => {
      opt.classList.toggle("active", opt.dataset.mode === currentMode);
    });
  }

  // ---- Chat Logic ----

  async function handleSend() {
    const text = dom.chatInput.value.trim();
    if (!text || isStreaming) return;

    // Ensure we have a session
    if (!activeSessionId) {
      createSession();
    }

    const session = getActiveSession();
    if (!session) return;

    // Add user message as tree node
    const parentId = session.activeLeafId || null;
    const userNode = addChildNode(session, parentId, "user", text);
    session.activeLeafId = userNode.id;
    updateSessionTitle(session.id);
    saveSessions();

    // Clear input
    dom.chatInput.value = "";
    dom.chatInput.style.height = "auto";

    // Render user message
    // Remove welcome screen if present
    const welcome = dom.messagesInner.querySelector(".welcome");
    if (welcome) welcome.remove();
    dom.messagesInner.appendChild(createMessageEl(userNode, session));
    userScrolledUp = false;
    scrollToBottom(true);

    // Start streaming response
    await sendMessage(session);
  }

  async function sendMessage(session) {
    isStreaming = true;
    updateSendButton();

    // Build messages array for the API from active tree path
    const messages = buildMessagesForAPI(session);

    // Add streaming message placeholder
    dom.messagesInner.appendChild(createStreamingMessageEl());
    scrollToBottom(true);

    abortController = new AbortController();

    try {
      const base = appSettings.backendUrl || "";

      // Determine model to use — apply mode prefix if set
      let model = selectedModel || "default";
      if (currentMode === "analytical" && !model.startsWith("a/")) {
        model = "a/" + model;
      } else if (currentMode === "narrative" && !model.startsWith("n/")) {
        model = "n/" + model;
      }

      // Apply context limit — keep the most recent N messages
      let contextMessages = messages;
      if (appSettings.contextLimit != null && appSettings.contextLimit > 0 && messages.length > appSettings.contextLimit) {
        contextMessages = messages.slice(-appSettings.contextLimit);
      }

      // Prepend system prompt if configured
      const apiMessages = appSettings.systemPrompt
        ? [{ role: "system", content: appSettings.systemPrompt }, ...contextMessages]
        : contextMessages;

      // Build options from sampling settings (only include non-null values)
      const options = {};
      if (appSettings.temperature != null) options.temperature = appSettings.temperature;
      if (appSettings.topP != null) options.top_p = appSettings.topP;
      if (appSettings.maxTokens != null) options.num_predict = appSettings.maxTokens;
      if (appSettings.frequencyPenalty != null) options.repeat_penalty = appSettings.frequencyPenalty;
      if (appSettings.presencePenalty != null) options.presence_penalty = appSettings.presencePenalty;
      if (appSettings.seed != null) options.seed = appSettings.seed;
      if (appSettings.stopSequences) {
        const stops = appSettings.stopSequences.split(",").map((s) => s.trim()).filter(Boolean);
        if (stops.length > 0) options.stop = stops;
      }

      const requestBody = { model: model, messages: apiMessages, stream: true };
      if (Object.keys(options).length > 0) requestBody.options = options;

      const response = await fetch(`${base}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
        signal: abortController.signal,
      });

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(errBody.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // Keep incomplete line in buffer

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            // Handle Augmentum phase metadata (UARF reasoning)
            if (data.augmentum) {
              handleAugmentumMeta(data.augmentum);
            }
            if (data.message && data.message.content) {
              appendToStreamingMessage(data.message.content);
            }
            if (data.done) {
              // Stream is complete
            }
          } catch {
            // Skip malformed lines
          }
        }
      }

      // Process any remaining buffer
      if (buffer.trim()) {
        try {
          const data = JSON.parse(buffer);
          if (data.augmentum) {
            handleAugmentumMeta(data.augmentum);
          }
          if (data.message && data.message.content) {
            appendToStreamingMessage(data.message.content);
          }
        } catch {
          // Skip
        }
      }
    } catch (err) {
      if (err.name === "AbortError") {
        // User stopped generation
      } else {
        appendToStreamingMessage(
          "\n\n*Error: " + escapeHtml(err.message) + "*"
        );
      }
    }

    // Finalize
    const content = finalizeStreamingMessage(session);
    if (content) {
      const assistantNode = addChildNode(session, session.activeLeafId, "assistant", content);
      session.activeLeafId = assistantNode.id;

      // Persist UARF reasoning metadata so it survives branch switches and reloads
      if (streamPhases.length > 0 || streamToolCalls.length > 0) {
        assistantNode.reasoning = {
          phases: streamPhases.map((p) => ({ name: p.name, status: p.status, output: p.output || "" })),
          complexity: streamComplexity || "",
          phaseContent: Object.assign({}, streamPhaseContent),
          toolCalls: streamToolCalls.map((tc) => ({
            tool: tc.tool,
            phase: tc.phase,
            success: tc.success,
            input: tc.input,
            output: tc.output,
          })),
        };
      }

      // Set node ID on DOM element and regenerate button for branch operations
      const streamEl = dom.messagesInner.querySelector(`.message[data-node-id="pending"]`);
      if (streamEl) {
        streamEl.dataset.nodeId = assistantNode.id;
        // Update all pending data-node-id attributes on action buttons
        streamEl.querySelectorAll(`[data-node-id="pending"]`).forEach((btn) => {
          btn.dataset.nodeId = assistantNode.id;
        });
      }
      saveSessions();
    }

    isStreaming = false;
    abortController = null;
    updateSendButton();

    // Switch back to editor view after streaming completes
    if (flowEditorMode === "live") {
      // Brief delay so user can see final state
      setTimeout(() => switchToEditorView(), 1500);
    }

    // Narrative state will update via event from narrative/index.js poller
  }

  function stopStreaming() {
    if (abortController) {
      abortController.abort();
    }
  }

  async function regenerateMessage(nodeId) {
    if (isStreaming) {
      stopStreaming();
      // Wait a tick for the abort to settle
      await new Promise((r) => setTimeout(r, 100));
    }

    const session = getActiveSession();
    if (!session || !sessionHasMessages(session)) return;

    const node = session.tree[nodeId];
    if (!node || node.role !== "assistant") return;

    // Set activeLeafId to the parent (user message) so sendMessage creates a sibling
    const parentId = node.parentId;
    if (!parentId) return;
    session.activeLeafId = parentId;
    saveSessions();

    // Remove the assistant message DOM element being regenerated
    const msgEl = dom.messagesInner.querySelector(`.message[data-node-id="${nodeId}"]`);
    if (msgEl) msgEl.remove();

    // Re-send to get a fresh sibling response
    await sendMessage(session);
  }

  // Legacy wrapper for any remaining references
  async function regenerateLastMessage() {
    const session = getActiveSession();
    if (!session || !session.activeLeafId) return;
    const leaf = session.tree[session.activeLeafId];
    if (leaf && leaf.role === "assistant") {
      await regenerateMessage(leaf.id);
    }
  }

  function updateRegenerateButtons() {
    // Only enable the regenerate button on the LAST assistant message on the active path
    const allAssistant = dom.messagesInner.querySelectorAll(".message.assistant");
    allAssistant.forEach((msg, idx) => {
      const regenBtn = msg.querySelector(".regenerate-btn");
      if (!regenBtn) return;
      if (idx === allAssistant.length - 1) {
        regenBtn.disabled = false;
      } else {
        regenBtn.disabled = true;
      }
    });
  }

  // ---- Edit Message ----

  function startEditMessage(nodeId) {
    const session = getActiveSession();
    if (!session) return;
    const node = session.tree[nodeId];
    if (!node || node.role !== "user") return;

    const msgEl = dom.messagesInner.querySelector(`.message[data-node-id="${nodeId}"]`);
    if (!msgEl) return;

    const bodyEl = msgEl.querySelector(".message-body") || msgEl;
    const contentEl = bodyEl.querySelector(".message-content");
    if (!contentEl) return;

    // Replace content with edit textarea
    contentEl.innerHTML = `
      <textarea class="edit-textarea">${escapeHtml(node.content)}</textarea>
      <div class="edit-actions">
        <button class="edit-save-btn" data-action="save-edit" data-node-id="${nodeId}">${icons.check}<span>Save &amp; Submit</span></button>
        <button class="edit-cancel-btn" data-action="cancel-edit">Cancel</button>
      </div>
    `;

    // Focus textarea and move cursor to end
    const textarea = contentEl.querySelector(".edit-textarea");
    if (textarea) {
      textarea.focus();
      textarea.setSelectionRange(textarea.value.length, textarea.value.length);
    }

    // Hide action buttons while editing
    const actionsEl = msgEl.querySelector(".message-actions");
    if (actionsEl) actionsEl.style.display = "none";
  }

  async function submitEditMessage(nodeId, newContent) {
    const session = getActiveSession();
    if (!session) return;
    const node = session.tree[nodeId];
    if (!node) return;

    // Create new sibling user node under the same parent
    const parentId = node.parentId;
    const newUserNode = addChildNode(session, parentId, "user", newContent);
    session.activeLeafId = newUserNode.id;
    saveSessions();

    // Re-render messages to show the new branch
    renderMessages();

    // Send to get a response
    await sendMessage(session);
  }

  // ---- Delete Message ----

  function deleteMessageNode(nodeId) {
    const session = getActiveSession();
    if (!session) return;
    const node = session.tree[nodeId];
    if (!node) return;

    // Image nodes: remove directly, no confirmation needed
    if (node.role === "image") {
      removeNodeAndDescendants(session, nodeId);
      saveSessions();
      renderMessages();
      return;
    }

    // Check if this is the active leaf
    const isActiveLeaf = session.activeLeafId === nodeId;
    const descendantCount = countDescendants(session, nodeId);

    // Non-leaf nodes with descendants: confirm before deleting
    if (!isActiveLeaf && descendantCount > 0) {
      if (!confirm(
        "This will delete this message and " + descendantCount +
        " descendant message(s). This cannot be undone. Continue?"
      )) {
        return;
      }
    }

    const parentId = node.parentId;
    removeNodeAndDescendants(session, nodeId);

    // Update activeLeafId
    if (parentId && session.tree[parentId]) {
      session.activeLeafId = getDeepestLeaf(session, parentId);
    } else {
      session.activeLeafId = null;
      session.rootId = null;
    }

    saveSessions();
    renderMessages();
  }

  // ---- Inline Image Generation from Message ----

  async function handleGenerateImageFromMessage(nodeId) {
    const session = getActiveSession();
    if (!session) return;
    const node = session.tree[nodeId];
    if (!node) return;

    const prompt = node.content;
    if (!prompt || !prompt.trim()) return;

    // Read image settings from sidebar panel
    const negEl = $("#img-negative");
    const negative = negEl ? negEl.value : "";
    const presetEl = $("#img-preset");
    const preset = presetEl ? presetEl.value : "";
    const widthEl = $("#img-width");
    const width = widthEl ? parseInt(widthEl.value) || 512 : 512;
    const heightEl = $("#img-height");
    const height = heightEl ? parseInt(heightEl.value) || 512 : 512;
    const stepsEl = $("#img-steps");
    const steps = stepsEl ? parseInt(stepsEl.value) || 20 : 20;
    const cfgEl = $("#img-cfg");
    const cfg = cfgEl ? parseFloat(cfgEl.value) || 7.0 : 7.0;
    const seedEl = $("#img-seed");
    const seed = seedEl ? parseInt(seedEl.value) || -1 : -1;
    const samplerEl = $("#img-sampler");
    const sampler = samplerEl ? samplerEl.value : "";
    const modelEl = $("#img-model");
    const model = modelEl ? modelEl.value : "";

    // Show loading state on the button
    const msgEl = dom.messagesInner.querySelector(
      `.message[data-node-id="${nodeId}"]`
    );
    const btn = msgEl ? msgEl.querySelector(".generate-img-btn") : null;
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = '<span class="btn-spinner"></span><span>Generating\u2026</span>';
    }

    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(base + "/api/image/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt.substring(0, 10000),
          negative_prompt: negative,
          preset: preset,
          width: width,
          height: height,
          steps: steps,
          cfg_scale: cfg,
          seed: seed,
          sampler: sampler || undefined,
          model: model,
        }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: "Generation failed" }));
        showToast(err.error || "Image generation failed", "error");
        return;
      }

      const data = await resp.json();

      // Create image node as child of the source message
      const imageNode = addChildNode(session, nodeId, "image", "");
      imageNode.imageData = {
        image_id: data.image_id,
        url: data.url || "/api/image/" + data.image_id,
        prompt: data.prompt || prompt.substring(0, 120),
        seed: data.seed,
        width: data.width,
        height: data.height,
        steps: data.steps,
        model: data.model || model,
        negative_prompt: negative,
      };
      // Do NOT update activeLeafId — image nodes are not on the conversation path
      saveSessions();

      // Insert inline image element directly after the source message in DOM
      const inlineEl = createInlineImageEl(imageNode);
      if (msgEl) {
        // Insert after the message and any existing inline images
        let insertBefore = msgEl.nextSibling;
        while (insertBefore && insertBefore.classList &&
               insertBefore.classList.contains("inline-image")) {
          insertBefore = insertBefore.nextSibling;
        }
        dom.messagesInner.insertBefore(inlineEl, insertBefore);
      } else {
        dom.messagesInner.appendChild(inlineEl);
      }
      scrollToBottom(true);

      // Refresh sidebar gallery
      refreshImageGallery();

    } catch (err) {
      showToast("Image generation failed: " + err.message, "error");
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = icons.image + "<span>Image</span>";
      }
    }
  }

  function createInlineImageEl(imgNode) {
    const data = imgNode.imageData || {};
    const base = appSettings.backendUrl || "";
    const fullUrl = data.url
      ? (data.url.startsWith("http") ? data.url : base + data.url)
      : "";

    const el = document.createElement("div");
    el.className = "inline-image";
    el.dataset.nodeId = imgNode.id;

    const promptSnippet = (data.prompt || "").substring(0, 80);
    const seedText = data.seed && data.seed !== -1 ? "Seed: " + data.seed : "";
    const dimsText = data.width && data.height ? data.width + "\u00d7" + data.height : "";
    const metaParts = [promptSnippet, seedText, dimsText].filter(Boolean);

    el.innerHTML = `
      <div class="inline-image-thumb">
        <img src="${escapeHtml(fullUrl)}" alt="${escapeHtml(promptSnippet)}" loading="lazy">
      </div>
      <div class="inline-image-info">
        <div class="inline-image-meta">${escapeHtml(metaParts.join(" | "))}</div>
        <div class="inline-image-actions">
          <button class="message-action-btn download-img-btn" data-action="download-image" data-image-id="${escapeHtml(data.image_id || "")}" title="Download PNG">${icons.download}<span>Save</span></button>
          <button class="message-action-btn delete-msg-btn" data-action="delete-message" data-node-id="${imgNode.id}" title="Remove image">${icons.trash}<span>Delete</span></button>
        </div>
      </div>
    `;

    // Click thumbnail to open lightbox
    const thumb = el.querySelector(".inline-image-thumb");
    if (thumb) {
      thumb.addEventListener("click", () => {
        openLightbox({
          image_id: data.image_id || "",
          prompt: data.prompt || "",
          seed: data.seed || -1,
          width: data.width || 0,
          height: data.height || 0,
          negative_prompt: data.negative_prompt || "",
          steps: data.steps || 0,
          cfg_scale: data.cfg_scale || 0,
          model: data.model || "",
        }, fullUrl);
      });
    }

    return el;
  }

  async function downloadImage(imageId) {
    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(`${base}/api/image/${imageId}`);
      if (!resp.ok) throw new Error("Failed to fetch image");
      const blob = await resp.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `augmentum-${imageId}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(a.href);
    } catch (err) {
      showToast("Download failed: " + err.message, "error");
    }
  }

  function updateSendButton() {
    const btnContainer = dom.sendBtn.parentElement;
    if (isStreaming) {
      // Replace send button with stop button
      dom.sendBtn.style.display = "none";
      let stopBtn = btnContainer.querySelector(".stop-btn");
      if (!stopBtn) {
        stopBtn = document.createElement("button");
        stopBtn.className = "stop-btn";
        stopBtn.title = "Stop generation";
        stopBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';
        stopBtn.addEventListener("click", stopStreaming);
        btnContainer.appendChild(stopBtn);
      }
      stopBtn.style.display = "flex";
    } else {
      dom.sendBtn.style.display = "flex";
      const stopBtn = btnContainer.querySelector(".stop-btn");
      if (stopBtn) stopBtn.style.display = "none";
    }
  }

  // ---- Narrative State (event-driven from narrative/index.js poller) ----

  function updateNarrativeIndicator(hasState, isExternal) {
    const btn = dom.toggleNarrativeBtn;
    const hadState = hasExternalNarrativeState;
    hasExternalNarrativeState = hasState && isExternal;

    // Add/remove notification dot for external narrative sessions
    let dot = btn.querySelector(".narrative-dot");
    if (hasState && isExternal) {
      if (!dot) {
        dot = document.createElement("span");
        dot.className = "narrative-dot";
        btn.style.position = "relative";
        btn.appendChild(dot);
      }
    } else if (dot) {
      dot.remove();
    }

    // Auto-open narrative panel when external state first detected
    if (hasState && isExternal && !hadState) {
      if (dom.narrativePanel.classList.contains("hidden")) {
        dom.narrativePanel.classList.remove("hidden");
        dom.toggleNarrativeBtn.classList.add("active");
      }
    }
  }

  function startNarrativePolling() {
    // Now handled by narrative/index.js — kept as no-op for compatibility
  }

  function stopNarrativePolling() {
    }
  }

  function renderNarrativeState(data, sessionId) {
    const content = dom.narrativeContent;
    if (!data.state) {
      const hint = currentMode !== "narrative"
        ? "Narrative state from external clients (e.g. SillyTavern) will appear here automatically."
        : "No narrative state available for this session.";
      content.innerHTML = `
        <div class="empty-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
          <p>${hint}</p>
        </div>
      `;
      return;
    }

    const state = data.state;
    let html = "";

    // Show external session banner when state comes from an API client
    if (sessionId && currentMode !== "narrative") {
      html += `<div class="external-session-banner">
        <span class="external-session-dot"></span>
        External session <code>${escapeHtml(sessionId.slice(0, 12))}</code>
      </div>`;
    }

    // Characters
    if (state.characters && state.characters.length > 0) {
      html += '<div class="state-section"><div class="state-section-title">Characters</div>';
      state.characters.forEach((c) => {
        html += `
          <div class="character-card">
            <div class="character-name">${escapeHtml(c.name || "Unknown")}</div>
            <div class="character-emotion">${escapeHtml(c.emotional_state || "neutral")}</div>
          </div>
        `;
      });
      html += "</div>";
    }

    // Scene
    if (state.scene) {
      const scene = state.scene;
      html += '<div class="state-section"><div class="state-section-title">Scene</div>';
      html += '<div class="scene-info">';
      if (scene.location) {
        html += `<div class="scene-row"><span class="scene-label">Location</span><span>${escapeHtml(scene.location)}</span></div>`;
      }
      if (scene.time_of_day) {
        html += `<div class="scene-row"><span class="scene-label">Time</span><span>${escapeHtml(scene.time_of_day)}</span></div>`;
      }
      if (scene.weather) {
        html += `<div class="scene-row"><span class="scene-label">Weather</span><span>${escapeHtml(scene.weather)}</span></div>`;
      }
      if (scene.atmosphere) {
        html += `<div class="scene-row"><span class="scene-label">Mood</span><span>${escapeHtml(scene.atmosphere)}</span></div>`;
      }
      html += "</div></div>";
    }

    // Plot Threads
    if (state.plots && state.plots.length > 0) {
      html += '<div class="state-section"><div class="state-section-title">Plot Threads</div>';
      state.plots.forEach((p) => {
        html += `
          <div class="plot-thread">
            <div class="plot-title">${escapeHtml(p.title || "Untitled Thread")}</div>
            <div class="plot-status">${escapeHtml(p.status || "active")}</div>
          </div>
        `;
      });
      html += "</div>";
    }

    // If we only have the external banner but no state sections, show waiting hint
    const hasStateSections = state.characters?.length || state.scene || state.plots?.length;
    if (!hasStateSections) {
      html += `
        <div class="empty-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
          <p>Session active — waiting for narrative state to populate.</p>
        </div>
      `;
    }

    content.innerHTML = html;
  }

  // ---- Augmentum Metadata Handler ----

  function handleAugmentumMeta(meta) {
    if (!meta) return;

    // Handle model thinking deltas (before other checks)
    if (meta.model_thinking_delta) {
      appendModelThinking(meta.model_thinking_delta);
      // Don't return — other metadata (mode, phases) may coexist
    }

    // Update current mode from stream metadata (once)
    if (meta.mode === "analytical" && currentMode !== "analytical") {
      currentMode = "analytical";
      updateModeBadge();

      // Auto-show reasoning panel when analytical mode detected
      if (dom.reasoningPanel.classList.contains("hidden")) {
        dom.reasoningPanel.classList.remove("hidden");
        dom.toggleReasoningBtn.classList.add("active");
        // Close narrative panel if open (unless external session is active)
        if (!hasExternalNarrativeState) {
          dom.narrativePanel.classList.add("hidden");
          dom.toggleNarrativeBtn.classList.remove("active");
        }
      }

      // Auto-switch to live view for analytical streaming
      if (flowEditorMode !== "live") {
        switchToLiveView(meta.flow_name || currentFlow?.name || "Flow");
      }
    }

    // Handle per-token phase content streaming (lightweight, no rebuild)
    if (meta.phase_content_delta) {
      appendPhaseContent(meta.phase, meta.phase_content_delta);
      return;
    }

    // Full phase status update (rebuilds thinking block)
    if (meta.phases) {
      renderReasoningPhases(meta.phases, meta.complexity, meta.confidence);
      updateStreamThinking(meta.phases, meta.complexity);
      // Feed live view when in analytical mode
      if (flowEditorMode === "live" || meta.mode === "analytical") {
        if (flowEditorMode !== "live") {
          switchToLiveView(meta.flow_name || "Flow");
        }
        updateLiveView(meta.phases, meta.complexity, meta.confidence);
      }
    }

    // Process tool calls from phase metadata
    if (meta.tool_calls && meta.tool_calls.length > 0) {
      for (const tc of meta.tool_calls) {
        addStreamToolCall(tc);
      }
    }
  }

  // ---- Reasoning Panel ----

  function renderReasoningPhases(phases, complexity, confidence) {
    const content = dom.reasoningContent;
    if (!phases || phases.length === 0) {
      content.innerHTML = `
        <div class="empty-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
          <p>Reasoning phases will appear here during analytical mode processing.</p>
        </div>
      `;
      return;
    }

    let html = "";

    // Complexity badge
    if (complexity) {
      const complexityClass = complexity === "simple" ? "badge-simple" : complexity === "moderate" ? "badge-moderate" : "badge-complex";
      html += `<div class="reasoning-header">`;
      html += `<span class="complexity-badge ${complexityClass}">${escapeHtml(complexity)}</span>`;
      if (confidence !== undefined && confidence !== null) {
        const pct = Math.round(confidence * 100);
        html += `<span class="confidence-badge">${pct}% confidence</span>`;
      }
      html += `</div>`;
    }

    phases.forEach((phase) => {
      const statusClass = phase.status || "pending";
      const icon =
        statusClass === "complete"
          ? "&#10003;"
          : statusClass === "running"
          ? "&#9679;"
          : "&#9675;";
      // Show phase content — prefer stored output (from persisted reasoning),
      // fall back to live stream state during active streaming
      const phaseContent = phase.output || streamPhaseContent[phase.name] || "";
      const hasContent = phaseContent.length > 0;
      const contentHtml = hasContent
        ? `<div class="phase-output expanded"><pre class="phase-output-pre">${escapeHtml(phaseContent)}</pre></div>`
        : "";
      const toggleHint = hasContent
        ? `<button class="phase-panel-toggle" data-action="toggle-panel-phase">\u25BC</button>`
        : "";
      html += `
        <div class="phase-item ${statusClass}">
          <div class="phase-icon ${statusClass}">${icon}</div>
          <div class="phase-info">
            <div class="phase-name">${escapeHtml(PHASE_DISPLAY_NAMES[phase.name] || phase.name)}${toggleHint}</div>
            ${contentHtml}
          </div>
        </div>
      `;
    });
    content.innerHTML = html;
  }

  function showDefaultReasoningPhases() {
    // Show flow editor instead of static phases
    if (flowEditorMode === "editor") return;
    renderReasoningPhases(
      ANALYTICAL_PHASES.map((name) => ({ name, status: "pending", output: "" }))
    );
  }

  // ---- Flow Editor ----

  async function loadFlows() {
    try {
      const resp = await fetch("/api/reasoning/flows");
      if (!resp.ok) return;
      flowList = await resp.json();
      renderFlowSelect();
      if (flowList.length > 0 && !currentFlow) {
        const def = flowList.find(f => f.is_default) || flowList[0];
        await selectFlow(def.id);
      }
    } catch (e) {
      console.warn("Failed to load flows:", e);
    }
  }

  function renderFlowSelect() {
    const sel = dom.flowSelect;
    sel.innerHTML = "";
    for (const f of flowList) {
      const opt = document.createElement("option");
      opt.value = f.id;
      let label = f.name;
      if (f.is_default) label += " (default)";
      else if (f.is_builtin) label += " *";
      opt.textContent = label;
      sel.appendChild(opt);
    }
    if (currentFlow) sel.value = currentFlow.id;
  }

  async function selectFlow(flowId) {
    try {
      const resp = await fetch(`/api/reasoning/flows/${flowId}`);
      if (!resp.ok) return;
      currentFlow = await resp.json();
      dom.flowSelect.value = flowId;
      renderFlowInfo();
      renderFlowSteps();
      hideStepEditor();
    } catch (e) {
      console.warn("Failed to load flow:", e);
    }
  }

  function renderFlowInfo() {
    if (!currentFlow) return;
    dom.flowInfoName.textContent = currentFlow.name;
    dom.flowInfoDesc.textContent = currentFlow.description || "";
    dom.flowBadgeDefault.classList.toggle("hidden", !currentFlow.is_default);
    dom.flowBadgeBuiltin.classList.toggle("hidden", !currentFlow.is_builtin);
    dom.flowDeleteBtn.classList.toggle("hidden", currentFlow.is_builtin);
    dom.flowSetDefaultBtn.classList.toggle("hidden", currentFlow.is_default);
    // Auto Routing is a meta-flow — hide step editing controls
    const isAutoRouting = currentFlow.name === "Auto Routing";
    dom.flowAddStepBtn.classList.toggle("hidden", isAutoRouting);
  }

  function renderFlowSteps() {
    if (!currentFlow) return;
    const list = dom.flowStepList;
    let html = "";
    currentFlow.steps.forEach((step, i) => {
      const roleTag = step.role !== "analyze" ? `<span class="flow-step-role-tag">${escapeHtml(step.role)}</span>` : "";
      const connector = i > 0 ? `<div class="flow-step-connector"></div>` : "";
      const disabledClass = step.enabled ? "" : " disabled";
      html += `
        <div class="flow-step-item${disabledClass}" data-step-index="${i}">
          ${connector}
          <span class="flow-step-drag" title="Drag to reorder">&#8942;&#8942;</span>
          <span class="flow-step-dot" data-role="${escapeHtml(step.role)}"></span>
          <span class="flow-step-name">${escapeHtml(step.name)}</span>
          ${roleTag}
        </div>`;
    });
    list.innerHTML = html;

    // Click to edit
    list.querySelectorAll(".flow-step-item").forEach(el => {
      el.addEventListener("click", () => {
        const idx = parseInt(el.dataset.stepIndex);
        openStepEditor(idx);
      });
    });

    // Drag to reorder
    initStepDragDrop();
  }

  function initStepDragDrop() {
    const list = dom.flowStepList;
    let dragItem = null;

    list.querySelectorAll(".flow-step-item").forEach(el => {
      el.draggable = true;
      el.addEventListener("dragstart", (e) => {
        dragItem = el;
        el.style.opacity = "0.4";
        e.dataTransfer.effectAllowed = "move";
      });
      el.addEventListener("dragend", () => {
        el.style.opacity = "";
        dragItem = null;
        list.querySelectorAll(".flow-step-item").forEach(s => s.classList.remove("drag-over"));
      });
      el.addEventListener("dragover", (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = "move";
        el.classList.add("drag-over");
      });
      el.addEventListener("dragleave", () => {
        el.classList.remove("drag-over");
      });
      el.addEventListener("drop", async (e) => {
        e.preventDefault();
        el.classList.remove("drag-over");
        if (!dragItem || dragItem === el) return;
        const fromIdx = parseInt(dragItem.dataset.stepIndex);
        const toIdx = parseInt(el.dataset.stepIndex);
        if (fromIdx === toIdx) return;
        // Reorder steps array
        const steps = [...currentFlow.steps];
        const [moved] = steps.splice(fromIdx, 1);
        steps.splice(toIdx, 0, moved);
        // Reassign sort_order
        steps.forEach((s, i) => s.sort_order = i);
        currentFlow.steps = steps;
        await saveFlowSteps();
        renderFlowSteps();
      });
    });
  }

  function openStepEditor(index) {
    if (!currentFlow || !currentFlow.steps[index]) return;
    editingStepIndex = index;
    const step = currentFlow.steps[index];

    // Store original for revert
    editingStepOriginal = JSON.parse(JSON.stringify(step));

    dom.flowStepsContainer.classList.add("hidden");
    dom.flowStepEditor.classList.remove("hidden");

    dom.stepNameInput.value = step.name;
    dom.stepRoleSelect.value = step.role || "analyze";
    dom.stepSystemPrompt.value = step.system_prompt || "";
    dom.stepUserTemplate.value = step.user_template || "";
    dom.stepOutputCap.value = step.output_cap || 800;
    dom.stepStreamToUser.checked = !!step.stream_to_user;

    // Complexity gate
    const gate = step.complexity_gate || [];
    dom.stepGateSimple.checked = gate.includes("simple");
    dom.stepGateModerate.checked = gate.includes("moderate");
    dom.stepGateComplex.checked = gate.includes("complex");

    // Tools grid
    renderToolsGrid(step);

    // All fields are always editable — built-in flows auto-clone on save
    dom.stepNameInput.disabled = false;
    dom.stepRoleSelect.disabled = false;
    dom.stepSystemPrompt.disabled = false;
    dom.stepUserTemplate.disabled = false;
    dom.stepOutputCap.disabled = false;
    dom.stepStreamToUser.disabled = false;
    dom.stepGateSimple.disabled = false;
    dom.stepGateModerate.disabled = false;
    dom.stepGateComplex.disabled = false;
    dom.stepSaveBtn.classList.remove("hidden");
    // Only show delete for non-builtin (clone gets its own steps after save)
    dom.stepDeleteBtn.classList.toggle("hidden", currentFlow.is_builtin);

    // Show revert buttons (always available — reverts to state when editor opened)
    dom.stepSystemPromptRevert.classList.remove("hidden");
    dom.stepUserTemplateRevert.classList.remove("hidden");
  }

  function renderToolsGrid(step) {
    const activeNames = new Set(step.tool_names || []);
    const activeCats = new Set(step.tool_categories || []);
    let html = "";

    // Category chips
    for (const cat of TOOL_CATEGORIES) {
      const active = activeCats.has(cat) ? " active" : "";
      html += `<span class="step-tool-chip${active}" data-type="category" data-value="${escapeHtml(cat)}">${escapeHtml(cat)}</span>`;
    }

    // Individual tool chips
    for (const tool of KNOWN_TOOLS) {
      const active = activeNames.has(tool) ? " active" : "";
      html += `<span class="step-tool-chip${active}" data-type="tool" data-value="${escapeHtml(tool)}">${escapeHtml(tool)}</span>`;
    }

    dom.stepToolsGrid.innerHTML = html;

    // Toggle clicks (always enabled — auto-clone on save handles built-ins)
    dom.stepToolsGrid.querySelectorAll(".step-tool-chip").forEach(chip => {
      chip.addEventListener("click", () => {
        chip.classList.toggle("active");
      });
    });
  }

  function hideStepEditor() {
    editingStepIndex = -1;
    editingStepOriginal = null;
    dom.flowStepEditor.classList.add("hidden");
    dom.flowStepsContainer.classList.remove("hidden");
    dom.stepSystemPromptRevert.classList.add("hidden");
    dom.stepUserTemplateRevert.classList.add("hidden");
  }

  function collectStepFromEditor() {
    if (editingStepIndex < 0 || !currentFlow) return null;
    const step = { ...currentFlow.steps[editingStepIndex] };

    step.name = dom.stepNameInput.value.trim() || "Untitled";
    step.role = dom.stepRoleSelect.value;
    step.system_prompt = dom.stepSystemPrompt.value;
    step.user_template = dom.stepUserTemplate.value;
    step.output_cap = parseInt(dom.stepOutputCap.value) || 0;
    step.stream_to_user = dom.stepStreamToUser.checked;

    // Complexity gate
    const gate = [];
    if (dom.stepGateSimple.checked) gate.push("simple");
    if (dom.stepGateModerate.checked) gate.push("moderate");
    if (dom.stepGateComplex.checked) gate.push("complex");
    step.complexity_gate = gate;

    // Tools
    const toolNames = [];
    const toolCats = [];
    dom.stepToolsGrid.querySelectorAll(".step-tool-chip.active").forEach(chip => {
      if (chip.dataset.type === "tool") toolNames.push(chip.dataset.value);
      else if (chip.dataset.type === "category") toolCats.push(chip.dataset.value);
    });
    step.tool_names = toolNames;
    step.tool_categories = toolCats;

    return step;
  }

  async function saveFlowSteps() {
    if (!currentFlow || currentFlow.is_builtin) return;
    try {
      const resp = await fetch(`/api/reasoning/flows/${currentFlow.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ steps: currentFlow.steps }),
      });
      if (resp.ok) {
        currentFlow = await resp.json();
        showToast("Flow saved", "success");
      } else {
        showToast("Failed to save flow", "error");
      }
    } catch (e) {
      showToast("Failed to save: " + e.message, "error");
    }
  }

  async function handleSaveStep() {
    const step = collectStepFromEditor();
    if (!step) return;

    // Auto-clone built-in flows on first edit
    if (currentFlow.is_builtin) {
      const name = currentFlow.name + " (custom)";
      try {
        const resp = await fetch(`/api/reasoning/flows/${currentFlow.id}/clone?name=${encodeURIComponent(name)}`, {
          method: "POST",
        });
        if (!resp.ok) { showToast("Failed to clone flow for editing", "error"); return; }
        const clone = await resp.json();
        // Apply the edited step to the clone
        clone.steps[editingStepIndex] = step;
        // Save the clone
        const saveResp = await fetch(`/api/reasoning/flows/${clone.id}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ steps: clone.steps }),
        });
        if (saveResp.ok) {
          await loadFlows();
          await selectFlow(clone.id);
          showToast(`Cloned as "${name}" and saved`, "success");
        }
      } catch (e) {
        showToast("Clone failed: " + e.message, "error");
      }
      return;
    }

    currentFlow.steps[editingStepIndex] = step;
    await saveFlowSteps();
    renderFlowSteps();
    hideStepEditor();
  }

  async function handleAddStep() {
    if (!currentFlow) return;

    const newStep = {
      name: "New Step",
      role: "analyze",
      system_prompt: "",
      user_template: "",
      tool_categories: [],
      tool_names: [],
      complexity_gate: [],
      stream_to_user: false,
      output_cap: 800,
      enabled: true,
      sort_order: currentFlow.steps.length,
    };

    // Auto-clone built-in flows when adding steps
    if (currentFlow.is_builtin) {
      const name = currentFlow.name + " (custom)";
      try {
        const resp = await fetch(`/api/reasoning/flows/${currentFlow.id}/clone?name=${encodeURIComponent(name)}`, {
          method: "POST",
        });
        if (!resp.ok) { showToast("Failed to clone flow", "error"); return; }
        const clone = await resp.json();
        clone.steps.push(newStep);
        const saveResp = await fetch(`/api/reasoning/flows/${clone.id}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ steps: clone.steps }),
        });
        if (saveResp.ok) {
          await loadFlows();
          await selectFlow(clone.id);
          showToast(`Cloned as "${name}"`, "success");
          openStepEditor(clone.steps.length - 1);
        }
      } catch (e) {
        showToast("Clone failed: " + e.message, "error");
      }
      return;
    }

    currentFlow.steps.push(newStep);
    await saveFlowSteps();
    renderFlowSteps();
    openStepEditor(currentFlow.steps.length - 1);
  }

  async function handleDeleteStep() {
    if (editingStepIndex < 0 || !currentFlow || currentFlow.is_builtin) return;
    if (currentFlow.steps.length <= 1) {
      showToast("Flow must have at least one step", "warning");
      return;
    }
    currentFlow.steps.splice(editingStepIndex, 1);
    currentFlow.steps.forEach((s, i) => s.sort_order = i);
    await saveFlowSteps();
    hideStepEditor();
    renderFlowSteps();
  }

  async function handleCloneFlow() {
    if (!currentFlow) return;
    const name = prompt("Name for the cloned flow:", currentFlow.name + " (custom)");
    if (!name) return;
    try {
      const resp = await fetch(`/api/reasoning/flows/${currentFlow.id}/clone?name=${encodeURIComponent(name)}`, {
        method: "POST",
      });
      if (resp.ok) {
        const clone = await resp.json();
        await loadFlows();
        await selectFlow(clone.id);
        showToast(`Cloned as "${name}"`, "success");
      }
    } catch (e) {
      showToast("Clone failed: " + e.message, "error");
    }
  }

  async function handleSetDefault() {
    if (!currentFlow) return;
    try {
      const resp = await fetch(`/api/reasoning/flows/${currentFlow.id}/default`, { method: "PUT" });
      if (resp.ok) {
        await loadFlows();
        await selectFlow(currentFlow.id);
        showToast("Set as default", "success");
      }
    } catch (e) {
      showToast("Failed: " + e.message, "error");
    }
  }

  async function handleDeleteFlow() {
    if (!currentFlow || currentFlow.is_builtin) return;
    if (!confirm(`Delete "${currentFlow.name}"?`)) return;
    try {
      const resp = await fetch(`/api/reasoning/flows/${currentFlow.id}`, { method: "DELETE" });
      if (resp.ok) {
        currentFlow = null;
        await loadFlows();
        showToast("Flow deleted", "success");
      }
    } catch (e) {
      showToast("Delete failed: " + e.message, "error");
    }
  }

  async function handleExportFlow() {
    if (!currentFlow) return;
    try {
      const resp = await fetch(`/api/reasoning/flows/${currentFlow.id}/export`);
      if (!resp.ok) return;
      const data = await resp.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${currentFlow.name.toLowerCase().replace(/\s+/g, "_")}_flow.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      showToast("Export failed: " + e.message, "error");
    }
  }

  async function handleImportFlow(file) {
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      const resp = await fetch("/api/reasoning/flows/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (resp.ok) {
        const imported = await resp.json();
        await loadFlows();
        await selectFlow(imported.id);
        showToast(`Imported "${imported.name}"`, "success");
      }
    } catch (e) {
      showToast("Import failed: " + e.message, "error");
    }
  }

  function showNewFlowModal() {
    // Fetch templates and show modal
    fetch("/api/reasoning/templates")
      .then(r => r.json())
      .then(templates => {
        let selectedTemplate = "";
        const overlay = document.createElement("div");
        overlay.className = "flow-new-modal-overlay";
        overlay.innerHTML = `
          <div class="flow-new-modal">
            <h3>New Reasoning Flow</h3>
            <div class="step-field">
              <label>Name</label>
              <input type="text" id="new-flow-name" placeholder="My Custom Flow" style="width:100%;padding:6px 8px;border-radius:4px;border:1px solid var(--border);background:var(--input-bg);color:var(--text);font-size:13px;">
            </div>
            <div class="step-field">
              <label>Start from template</label>
              <div class="flow-template-list">
                <div class="flow-template-item" data-template="">
                  <div class="flow-template-info">
                    <div class="flow-template-name">Blank Canvas</div>
                    <div class="flow-template-desc">Start with an empty step</div>
                  </div>
                  <div class="flow-template-steps">1 step</div>
                </div>
                ${templates.map(t => `
                  <div class="flow-template-item" data-template="${escapeHtml(t.name)}">
                    <div class="flow-template-info">
                      <div class="flow-template-name">${escapeHtml(t.display_name)}</div>
                      <div class="flow-template-desc">${escapeHtml(t.description)}</div>
                    </div>
                    <div class="flow-template-steps">${t.step_count} steps</div>
                  </div>
                `).join("")}
              </div>
            </div>
            <div class="flow-new-actions">
              <button class="btn btn-xs" id="new-flow-cancel">Cancel</button>
              <button class="btn btn-primary btn-sm" id="new-flow-create">Create</button>
            </div>
          </div>`;

        document.body.appendChild(overlay);

        // Template selection
        overlay.querySelectorAll(".flow-template-item").forEach(el => {
          el.addEventListener("click", () => {
            overlay.querySelectorAll(".flow-template-item").forEach(x => x.classList.remove("selected"));
            el.classList.add("selected");
            selectedTemplate = el.dataset.template;
          });
        });

        // Select first by default (blank)
        overlay.querySelector(".flow-template-item").classList.add("selected");

        overlay.querySelector("#new-flow-cancel").addEventListener("click", () => overlay.remove());
        overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });

        overlay.querySelector("#new-flow-create").addEventListener("click", async () => {
          const name = overlay.querySelector("#new-flow-name").value.trim();
          if (!name) {
            showToast("Please enter a name", "warning");
            return;
          }
          const body = { name };
          if (selectedTemplate) {
            body.template = selectedTemplate;
          } else {
            body.steps = [{ name: "Respond", role: "respond", system_prompt: "Answer the user's question.", stream_to_user: true, output_cap: 0 }];
          }
          try {
            const resp = await fetch("/api/reasoning/flows", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(body),
            });
            if (resp.ok) {
              const created = await resp.json();
              overlay.remove();
              await loadFlows();
              await selectFlow(created.id);
              showToast(`Created "${name}"`, "success");
            }
          } catch (e) {
            showToast("Create failed: " + e.message, "error");
          }
        });
      })
      .catch(e => showToast("Failed to load templates", "error"));
  }

  // Live execution view integration
  function switchToLiveView(flowName) {
    flowEditorMode = "live";
    dom.reasoningEditorView.classList.add("hidden");
    dom.reasoningLiveView.classList.remove("hidden");
    dom.reasoningContent.classList.add("hidden");
    dom.reasoningPanelTitle.textContent = "Running";
    dom.liveFlowName.textContent = flowName || "Flow";
    dom.liveFlowComplexity.textContent = "";
    dom.liveStepList.innerHTML = "";
    dom.liveStats.innerHTML = "";
  }

  function switchToEditorView() {
    flowEditorMode = "editor";
    dom.reasoningEditorView.classList.remove("hidden");
    dom.reasoningLiveView.classList.add("hidden");
    dom.reasoningContent.classList.add("hidden");
    dom.reasoningPanelTitle.textContent = "Reasoning Flows";
  }

  function updateLiveView(phases, complexity, confidence) {
    if (flowEditorMode !== "live") return;
    if (complexity) {
      dom.liveFlowComplexity.textContent = complexity;
    }
    if (!phases) return;
    let html = "";
    let toolsUsed = 0;
    let completedCount = 0;

    phases.forEach(p => {
      let icon, cls;
      if (p.status === "complete") {
        icon = "&#10003;";
        cls = "complete";
        completedCount++;
      } else if (p.status === "running") {
        icon = "&#9679;";
        cls = "running";
      } else if (p.status === "skipped") {
        icon = "&#8212;";
        cls = "skipped";
      } else {
        icon = "&#9675;";
        cls = "pending";
      }
      html += `
        <div class="live-step-item ${cls}">
          <span class="live-step-icon">${icon}</span>
          <span class="live-step-name">${escapeHtml(p.name)}</span>
        </div>`;
    });
    dom.liveStepList.innerHTML = html;

    // Stats
    let stats = `${completedCount}/${phases.length} steps`;
    if (complexity) stats += ` | ${complexity}`;
    if (confidence !== undefined && confidence !== null) stats += ` | ${Math.round(confidence * 100)}% confidence`;
    dom.liveStats.innerHTML = stats;
  }

  // ---- Markdown Renderer ----

  function renderMarkdown(text) {
    if (!text) return "";

    // Step 1: Escape HTML to prevent XSS
    let html = escapeHtml(text);

    // Sentinel for placeholders. \x01 (SOH) never appears in user content and
    // survives DOM serialization, so the underscore-emphasis pass below can't
    // accidentally devour `__CODE_BLOCK_0__`-shaped tokens.
    const SE = "\x01";

    // Step 2: Code blocks (``` ... ```) — must be done first to protect content inside
    const codeBlocks = [];
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
      const placeholder = `${SE}CB${codeBlocks.length}${SE}`;
      const rawCode = unescapeHtml(code.trimEnd());
      const langClass = lang ? ` class="language-${escapeHtml(lang)}"` : "";
      const langLabel = lang ? `<div class="code-header"><span>${escapeHtml(lang)}</span><button class="copy-btn" data-copy="${encodeURIComponent(rawCode)}">Copy</button></div>` : "";
      codeBlocks.push(
        `${langLabel}<pre><code${langClass}>${code.trimEnd()}</code></pre>`
      );
      return placeholder;
    });

    // Inline code (must be before bold/italic to avoid conflicts)
    const inlineCodes = [];
    html = html.replace(/`([^`\n]+)`/g, (match, code) => {
      const placeholder = `${SE}IC${inlineCodes.length}${SE}`;
      inlineCodes.push(`<code>${code}</code>`);
      return placeholder;
    });

    // Smart typography on prose-only (code is sentineled). Non-dash lookarounds
    // keep `----` (HR runs) intact for the HR pass below.
    html = html.replace(/(?<=[^\s-])--(?=[^\s-])/g, "—");
    html = html.replace(/\.\.\./g, "…");

    // Horizontal rules — three or more of -, *, or _ on their own line.
    html = html.replace(/^[ ]{0,3}-(?:[ ]*-){2,}[ ]*$/gm, "<hr>");
    html = html.replace(/^[ ]{0,3}\*(?:[ ]*\*){2,}[ ]*$/gm, "<hr>");
    html = html.replace(/^[ ]{0,3}_(?:[ ]*_){2,}[ ]*$/gm, "<hr>");

    // Headings — H1 through H6 (longest first).
    html = html.replace(/^###### (.+)$/gm, "<h6>$1</h6>");
    html = html.replace(/^##### (.+)$/gm, "<h5>$1</h5>");
    html = html.replace(/^#### (.+)$/gm, "<h4>$1</h4>");
    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");

    // Bold — `**X**` and `__X__` (underscore form word-boundary-guarded so
    // `my__var__name` stays literal).
    html = html.replace(/\*\*(?!\s)([\s\S]+?)(?<!\s)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/(^|[^\w])__(?!\s)([\s\S]+?)(?<!\s)__(?!\w)/g, "$1<strong>$2</strong>");

    // Italic — `*X*` (non-whitespace boundaries; ignores `2 * 3` math and
    // `* ` list markers) and `_X_` (word-boundary-guarded for snake_case).
    html = html.replace(/(?<![*\w])\*(?!\s|\*)([\s\S]+?)(?<!\s|\*)\*(?![*\w])/g, "<em>$1</em>");
    html = html.replace(/(^|[^\w])_(?!\s)([\s\S]+?)(?<!\s)_(?!\w)/g, "$1<em>$2</em>");

    // Blockquotes — merge consecutive `&gt; ` lines into one quote block.
    html = html.replace(/(?:^&gt; [^\n]*(?:\n|$))+/gm, (block) => {
      const inner = block
        .replace(/\n$/, "")
        .split("\n")
        .map((l) => l.replace(/^&gt; /, ""))
        .join("<br>");
      return `<blockquote>${inner}</blockquote>\n`;
    });

    // Lists — accept `-`, `*`, `+` as unordered bullets; tag UL vs OL with
    // sentinels so the wrap pass keeps them in distinct list elements.
    html = html.replace(/^[ ]{0,3}[-*+] (.+)$/gm, `${SE}UL${SE}<li>$1</li>`);
    html = html.replace(/^[ ]{0,3}\d+\. (.+)$/gm, `${SE}OL${SE}<li>$1</li>`);
    html = html.replace(new RegExp(`(?:${SE}UL${SE}<li>[^\\n]*</li>\\n?)+`, "g"),
      (m) => `<ul>${m.split(`${SE}UL${SE}`).join("")}</ul>`);
    html = html.replace(new RegExp(`(?:${SE}OL${SE}<li>[^\\n]*</li>\\n?)+`, "g"),
      (m) => `<ol>${m.split(`${SE}OL${SE}`).join("")}</ol>`);

    // Line breaks — convert double newlines to paragraph breaks, single to <br>
    html = html.replace(/\n\n/g, "</p><p>");
    html = html.replace(/\n/g, "<br>");

    // Wrap in paragraph
    html = "<p>" + html + "</p>";

    // Clean up empty paragraphs
    html = html.replace(/<p><\/p>/g, "");
    html = html.replace(/<p>(<h[1-6]>)/g, "$1");
    html = html.replace(/(<\/h[1-6]>)<\/p>/g, "$1");
    html = html.replace(/<p>(<ul>)/g, "$1");
    html = html.replace(/(<\/ul>)<\/p>/g, "$1");
    html = html.replace(/<p>(<ol>)/g, "$1");
    html = html.replace(/(<\/ol>)<\/p>/g, "$1");
    html = html.replace(/<p>(<blockquote>)/g, "$1");
    html = html.replace(/(<\/blockquote>)<\/p>/g, "$1");
    html = html.replace(/<p>(<pre>)/g, "$1");
    html = html.replace(/(<\/pre>)<\/p>/g, "$1");
    html = html.replace(/<p>(<hr>)<\/p>/g, "$1");
    html = html.replace(/<p>(<hr>)/g, "$1");
    html = html.replace(/(<hr>)<\/p>/g, "$1");
    html = html.replace(/<p>(<div class="code-header">)/g, "$1");

    // Restore code blocks
    codeBlocks.forEach((block, i) => {
      html = html.replace(`${SE}CB${i}${SE}`, block);
    });

    // Restore inline codes
    inlineCodes.forEach((code, i) => {
      html = html.replace(`${SE}IC${i}${SE}`, code);
    });

    return html;
  }

  /**
   * Apply syntax highlighting to all code blocks in a container.
   * Uses highlight.js if available, gracefully degrades if CDN fails.
   */
  function highlightCode(container) {
    if (typeof hljs === "undefined") return;
    const blocks = (container || document).querySelectorAll("pre code[class*='language-']");
    blocks.forEach((block) => {
      if (!block.dataset.highlighted) {
        hljs.highlightElement(block);
      }
    });
  }

  // ---- Toast Notifications ----

  const TOAST_ICONS = {
    success: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>',
    warning: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    error: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
    info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
  };

  /**
   * Show a toast notification.
   * @param {string} message - The message text
   * @param {"success"|"warning"|"error"|"info"} type - Toast type
   * @param {number} duration - Auto-dismiss after ms (0 = manual only)
   */
  function showToast(message, type = "info", duration = 4000) {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const el = document.createElement("div");
    el.className = `toast toast-${type}`;
    el.innerHTML = `
      <span class="toast-icon">${TOAST_ICONS[type] || TOAST_ICONS.info}</span>
      <span>${escapeHtml(message)}</span>
      <button class="toast-close" aria-label="Dismiss">&times;</button>
    `;

    const dismiss = () => {
      el.classList.add("toast-exiting");
      el.addEventListener("animationend", () => el.remove());
    };
    el.querySelector(".toast-close").addEventListener("click", dismiss);

    container.appendChild(el);

    if (duration > 0) {
      setTimeout(dismiss, duration);
    }
  }

  // ---- Utility Functions ----

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  function unescapeHtml(str) {
    const doc = new DOMParser().parseFromString(str, "text/html");
    return doc.documentElement.textContent;
  }

  /**
   * Build an expandable/collapsible section for long content.
   * Shows a preview (first 200 chars) with a "Show more" toggle when
   * content exceeds the threshold, or the full content when it's short.
   */
  function buildExpandableSection(label, content, previewLen) {
    previewLen = previewLen || 200;
    const escaped = escapeHtml(content);
    const needsExpand = content.length > previewLen;

    if (!needsExpand) {
      return `<div class="tool-call-section">
        <div class="tool-call-section-label">${escapeHtml(label)}</div>
        <pre class="tool-call-pre">${escaped}</pre>
      </div>`;
    }

    const preview = escapeHtml(content.slice(0, previewLen)) + "\u2026";
    return `<div class="tool-call-section expandable-section">
      <div class="tool-call-section-label">${escapeHtml(label)}</div>
      <pre class="tool-call-pre expandable-preview">${preview}</pre>
      <pre class="tool-call-pre expandable-full" style="display:none">${escaped}</pre>
      <button class="expand-toggle-btn" data-action="expand-section">Show more</button>
    </div>`;
  }

  function autoGrowTextarea(el) {
    el.addEventListener("input", () => {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 160) + "px";
    });
  }

  // ---- Slider Sync Helper ----

  function syncSlider(slider, numberInput) {
    slider.addEventListener("input", () => {
      numberInput.value = slider.value;
    });
    numberInput.addEventListener("input", () => {
      const val = parseFloat(numberInput.value);
      if (!isNaN(val)) {
        slider.value = val;
      }
    });
    // Clear number input clears slider to default min
    numberInput.addEventListener("change", () => {
      if (numberInput.value === "") {
        slider.value = slider.min;
      }
    });
  }

  // ---- Event Binding ----

  function closeSidebar() {
    dom.sidebar.classList.add("collapsed");
    dom.sidebarOverlay.classList.remove("visible");
    appSettings.sidebarOpen = false;
    saveSettings();
  }

  function toggleSidebar() {
    dom.sidebar.classList.toggle("collapsed");
    dom.sidebarOverlay.classList.toggle("visible", !dom.sidebar.classList.contains("collapsed"));
    appSettings.sidebarOpen = !dom.sidebar.classList.contains("collapsed");
    saveSettings();
  }

  function bindEvents() {
    // Sidebar toggle / close
    dom.toggleSidebarBtn.addEventListener("click", toggleSidebar);
    dom.sidebarCloseBtn.addEventListener("click", closeSidebar);
    dom.sidebarOverlay.addEventListener("click", closeSidebar);

    // New chat
    dom.newChatBtn.addEventListener("click", createSession);

    // Send
    dom.sendBtn.addEventListener("click", handleSend);
    dom.chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    });

    // Model selector
    dom.modelSelector.addEventListener("change", (e) => {
      selectedModel = e.target.value;
    });

    // Panel toggles
    dom.toggleReasoningBtn.addEventListener("click", () => {
      const panel = dom.reasoningPanel;
      const isHidden = panel.classList.contains("hidden");
      panel.classList.toggle("hidden");
      dom.toggleReasoningBtn.classList.toggle("active", isHidden);
      // Close narrative panel if opening reasoning (unless external session active)
      if (isHidden && !hasExternalNarrativeState) {
        dom.narrativePanel.classList.add("hidden");
        dom.toggleNarrativeBtn.classList.remove("active");
      }
      if (isHidden) {
        switchToEditorView();
      }
    });

    dom.toggleNarrativeBtn.addEventListener("click", () => {
      const panel = dom.narrativePanel;
      const isHidden = panel.classList.contains("hidden");
      panel.classList.toggle("hidden");
      dom.toggleNarrativeBtn.classList.toggle("active", isHidden);
      // Close reasoning panel if opening narrative
      if (isHidden) {
        dom.reasoningPanel.classList.add("hidden");
        dom.toggleReasoningBtn.classList.remove("active");
      }
    });

    dom.closeReasoningBtn.addEventListener("click", () => {
      dom.reasoningPanel.classList.add("hidden");
      dom.toggleReasoningBtn.classList.remove("active");
    });

    // Flow editor events
    if (dom.flowSelect) dom.flowSelect.addEventListener("change", (e) => {
      if (e.target.value) selectFlow(e.target.value);
    });
    if (dom.flowNewBtn) dom.flowNewBtn.addEventListener("click", showNewFlowModal);
    if (dom.flowCloneBtn) dom.flowCloneBtn.addEventListener("click", handleCloneFlow);
    if (dom.flowSetDefaultBtn) dom.flowSetDefaultBtn.addEventListener("click", handleSetDefault);
    if (dom.flowExportBtn) dom.flowExportBtn.addEventListener("click", handleExportFlow);
    if (dom.flowDeleteBtn) dom.flowDeleteBtn.addEventListener("click", handleDeleteFlow);
    if (dom.flowAddStepBtn) dom.flowAddStepBtn.addEventListener("click", handleAddStep);
    if (dom.stepEditorBack) dom.stepEditorBack.addEventListener("click", hideStepEditor);
    if (dom.stepSaveBtn) dom.stepSaveBtn.addEventListener("click", handleSaveStep);
    if (dom.stepDeleteBtn) dom.stepDeleteBtn.addEventListener("click", handleDeleteStep);
    if (dom.stepSystemPromptRevert) dom.stepSystemPromptRevert.addEventListener("click", () => {
      if (editingStepOriginal) {
        dom.stepSystemPrompt.value = editingStepOriginal.system_prompt || "";
        showToast("System prompt reverted", "success");
      }
    });
    if (dom.stepUserTemplateRevert) dom.stepUserTemplateRevert.addEventListener("click", () => {
      if (editingStepOriginal) {
        dom.stepUserTemplate.value = editingStepOriginal.user_template || "";
        showToast("User template reverted", "success");
      }
    });
    if (dom.flowImportBtn) dom.flowImportBtn.addEventListener("click", () => {
      if (dom.flowImportFile) dom.flowImportFile.click();
    });
    if (dom.flowImportFile) dom.flowImportFile.addEventListener("change", (e) => {
      if (e.target.files[0]) {
        handleImportFlow(e.target.files[0]);
        e.target.value = "";
      }
    });
    if (dom.reasoningViewToggle) dom.reasoningViewToggle.addEventListener("click", () => {
      if (flowEditorMode === "live") {
        switchToEditorView();
      } else {
        switchToLiveView(currentFlow ? currentFlow.name : "Flow");
      }
    });

    dom.closeNarrativeBtn.addEventListener("click", () => {
      dom.narrativePanel.classList.add("hidden");
      dom.toggleNarrativeBtn.classList.remove("active");
    });

    // Settings
    dom.settingsBtn.addEventListener("click", openSettings);
    dom.settingsClose.addEventListener("click", closeSettings);
    dom.settingsCancel.addEventListener("click", closeSettings);
    dom.settingsSave.addEventListener("click", saveSettingsFromModal);
    dom.settingsModal.addEventListener("click", (e) => {
      if (e.target === dom.settingsModal) closeSettings();
    });

    // Memory panel
    if (dom.memorySearchBtn) dom.memorySearchBtn.addEventListener("click", searchMemories);
    if (dom.memorySearchInput) dom.memorySearchInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") searchMemories();
    });
    if (dom.memoryAddBtn) dom.memoryAddBtn.addEventListener("click", addMemory);
    if (dom.memoryExportBtn) dom.memoryExportBtn.addEventListener("click", exportMemories);
    if (dom.memoryCompactBtn) dom.memoryCompactBtn.addEventListener("click", compactMemories);

    // MCP panel
    if (dom.mcpConnectBtn) dom.mcpConnectBtn.addEventListener("click", mcpConnectServer);

    // Mode switcher popup
    dom.modeBadge.addEventListener("click", (e) => {
      e.stopPropagation();
      const popup = dom.modePopup;
      const isOpen = popup.classList.contains("visible");
      popup.classList.toggle("visible", !isOpen);
      if (!isOpen) updateModePopupActiveState();
    });
    dom.modePopup.addEventListener("click", (e) => {
      e.stopPropagation();
      const opt = e.target.closest(".mode-option");
      if (!opt) return;
      const newMode = opt.dataset.mode;
      if (newMode === currentMode) {
        dom.modePopup.classList.remove("visible");
        return;
      }
      currentMode = newMode;
      appSettings.defaultMode = newMode;
      saveSettings();
      updateModeBadge();
      dom.modePopup.classList.remove("visible");
    });

    // Settings tab switching
    dom.settingsTabs.addEventListener("click", (e) => {
      const tab = e.target.closest(".settings-tab");
      if (!tab) return;
      const tabName = tab.dataset.tab;
      dom.settingsTabs.querySelectorAll(".settings-tab").forEach((t) =>
        t.classList.toggle("active", t.dataset.tab === tabName)
      );
      $$(".settings-tab-content").forEach((c) =>
        c.classList.toggle("active", c.id === "settings-tab-" + tabName)
      );
      // Load memory data when switching to memory tab
      if (tabName === "memory") {
        loadMemoryStats();
        loadMemoryList();
      }
      // Load MCP data when switching to MCP tab
      if (tabName === "mcp") {
        mcpLoadServers();
        mcpLoadTools();
      }
    });

    // Slider ↔ number input sync
    syncSlider(dom.settingTemperatureSlider, dom.settingTemperature);
    syncSlider(dom.settingTopPSlider, dom.settingTopP);
    syncSlider(dom.settingFreqPenaltySlider, dom.settingFreqPenalty);
    syncSlider(dom.settingPresPenaltySlider, dom.settingPresPenalty);

    // Tools tab slider sync
    syncSlider($("#setting-search-queries-slider"), $("#setting-search-queries"));
    syncSlider($("#setting-search-results-slider"), $("#setting-search-results"));
    syncSlider($("#setting-search-context-slider"), $("#setting-search-context"));
    syncSlider($("#setting-search-retries-slider"), $("#setting-search-retries"));
    syncSlider($("#setting-max-tool-calls-slider"), $("#setting-max-tool-calls"));

    // Theme toggle in settings
    $$(".theme-option").forEach((btn) => {
      btn.addEventListener("click", () => {
        applyTheme(btn.dataset.theme);
      });
    });

    // Shortcut chips on welcome screen
    $$(".shortcut-chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        dom.chatInput.value = btn.dataset.prompt;
        handleSend();
      });
    });

    // Model Manager
    dom.manageModelsBtn.addEventListener("click", openModelManager);
    dom.modelManagerClose.addEventListener("click", closeModelManager);
    dom.modelManagerModal.addEventListener("click", (e) => {
      if (e.target === dom.modelManagerModal) closeModelManager();
    });
    dom.mmPullBtn.addEventListener("click", () => {
      const backend = dom.mmBackendSelect.value;
      if (backend === "llamacpp") {
        fetchGgufFiles(dom.mmPullInput.value.trim());
      } else {
        pullModel(dom.mmPullInput.value);
      }
    });
    dom.mmPullInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        const backend = dom.mmBackendSelect.value;
        if (backend === "llamacpp") {
          fetchGgufFiles(dom.mmPullInput.value.trim());
        } else {
          pullModel(dom.mmPullInput.value);
        }
      }
    });
    dom.mmCancelBtn.addEventListener("click", () => {
      if (pullAbortController) {
        pullAbortController.abort();
      }
    });
    // Backend selector change
    dom.mmBackendSelect.addEventListener("change", () => {
      updateMmBackendUI();
    });
    updateMmBackendUI();

    // Provider Manager
    dom.settingsOpenProviders.addEventListener("click", () => {
      closeSettings();
      openProviderModal();
    });
    dom.providerModalClose.addEventListener("click", closeProviderModal);
    dom.providerModal.addEventListener("click", (e) => {
      if (e.target === dom.providerModal) closeProviderModal();
    });
    dom.provTestBtn.addEventListener("click", testProviderConnection);
    dom.provAddBtn.addEventListener("click", addProvider);

    // Delegated handlers for dynamic content (avoids inline onclick)
    document.addEventListener("click", (e) => {
      // Close mode popup on outside click
      if (dom.modePopup.classList.contains("visible") &&
          !e.target.closest(".mode-switcher")) {
        dom.modePopup.classList.remove("visible");
      }
      // Toggle parent open/close (thinking blocks, tool call cards)
      const toggleEl = e.target.closest("[data-toggle-parent]");
      if (toggleEl) {
        toggleEl.parentElement.classList.toggle(toggleEl.dataset.toggleParent);
        return;
      }
      // Expand/collapse toggle for long content sections
      const expandBtn = e.target.closest("[data-action='expand-section']");
      if (expandBtn) {
        const section = expandBtn.closest(".expandable-section");
        if (section) {
          const preview = section.querySelector(".expandable-preview");
          const full = section.querySelector(".expandable-full");
          if (preview && full) {
            const isExpanded = full.style.display !== "none";
            preview.style.display = isExpanded ? "" : "none";
            full.style.display = isExpanded ? "none" : "";
            expandBtn.textContent = isExpanded ? "Show more" : "Show less";
          }
        }
        return;
      }
      // Expand/collapse for thinking phase content
      const phaseToggle = e.target.closest("[data-action='toggle-phase-content']");
      if (phaseToggle) {
        // Button is inside .thinking-phase; content div is its next sibling at parent level
        const phaseDiv = phaseToggle.closest(".thinking-phase");
        const phaseContent = phaseDiv ? phaseDiv.nextElementSibling : null;
        if (phaseContent && phaseContent.classList.contains("thinking-phase-content")) {
          phaseContent.classList.toggle("collapsed");
          const icon = phaseToggle.querySelector(".phase-expand-icon");
          if (icon) icon.textContent = phaseContent.classList.contains("collapsed") ? "\u25B6" : "\u25BC";
        }
        return;
      }
      // Toggle phase output in reasoning side panel
      const panelPhaseToggle = e.target.closest("[data-action='toggle-panel-phase']");
      if (panelPhaseToggle) {
        const phaseInfo = panelPhaseToggle.closest(".phase-info");
        if (phaseInfo) {
          const output = phaseInfo.querySelector(".phase-output");
          if (output) {
            output.classList.toggle("expanded");
            panelPhaseToggle.textContent = output.classList.contains("expanded") ? "\u25BC" : "\u25B6";
          }
        }
        return;
      }
      // Code block copy buttons
      const copyEl = e.target.closest("[data-copy]");
      if (copyEl) {
        const text = decodeURIComponent(copyEl.dataset.copy);
        navigator.clipboard.writeText(text).then(() => {
          copyEl.textContent = "Copied!";
          setTimeout(() => { copyEl.textContent = "Copy"; }, 1500);
        });
        return;
      }
      // Branch navigation — previous sibling
      const branchPrev = e.target.closest("[data-action='branch-prev']");
      if (branchPrev) {
        const nid = branchPrev.dataset.nodeId;
        const session = getActiveSession();
        if (session && nid) {
          switchToSibling(session, nid, -1);
          saveSessions();
          renderMessages();
        }
        return;
      }
      // Branch navigation — next sibling
      const branchNext = e.target.closest("[data-action='branch-next']");
      if (branchNext) {
        const nid = branchNext.dataset.nodeId;
        const session = getActiveSession();
        if (session && nid) {
          switchToSibling(session, nid, +1);
          saveSessions();
          renderMessages();
        }
        return;
      }
      // Regenerate message (tree-aware)
      const regenAction = e.target.closest("[data-action='regenerate-message']");
      if (regenAction) {
        const nid = regenAction.dataset.nodeId;
        if (nid && nid !== "pending") {
          regenerateMessage(nid);
        }
        return;
      }
      // Edit user message
      const editAction = e.target.closest("[data-action='edit-message']");
      if (editAction) {
        const nid = editAction.dataset.nodeId;
        if (nid) startEditMessage(nid);
        return;
      }
      // Save edit
      const saveEdit = e.target.closest("[data-action='save-edit']");
      if (saveEdit) {
        const nid = saveEdit.dataset.nodeId;
        const msgEl = saveEdit.closest(".message");
        const textarea = msgEl ? msgEl.querySelector(".edit-textarea") : null;
        if (nid && textarea) {
          const newContent = textarea.value.trim();
          if (newContent) submitEditMessage(nid, newContent);
        }
        return;
      }
      // Cancel edit
      const cancelEdit = e.target.closest("[data-action='cancel-edit']");
      if (cancelEdit) {
        renderMessages(); // Re-render to restore original view
        return;
      }
      // Generate image from message
      const genImgAction = e.target.closest("[data-action='generate-image']");
      if (genImgAction) {
        const nid = genImgAction.dataset.nodeId;
        if (nid && nid !== "pending") {
          handleGenerateImageFromMessage(nid);
        }
        return;
      }
      // Inspect reasoning — load stored reasoning into inspector panel
      const inspectAction = e.target.closest("[data-action='inspect-reasoning']");
      if (inspectAction) {
        const nid = inspectAction.dataset.nodeId;
        if (nid) inspectMessageReasoning(nid);
        return;
      }
      // Delete message
      const deleteAction = e.target.closest("[data-action='delete-message']");
      if (deleteAction) {
        const nid = deleteAction.dataset.nodeId;
        if (nid) deleteMessageNode(nid);
        return;
      }
      // Download image
      const downloadAction = e.target.closest("[data-action='download-image']");
      if (downloadAction) {
        const imgId = downloadAction.dataset.imageId;
        if (imgId) downloadImage(imgId);
        return;
      }
    });

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      const mod = e.ctrlKey || e.metaKey;

      // Ctrl+Shift+N — New chat
      if (mod && e.shiftKey && e.key === "N") {
        e.preventDefault();
        createSession();
        return;
      }

      // Ctrl+/ — Focus chat input
      if (mod && e.key === "/") {
        e.preventDefault();
        dom.chatInput.focus();
        return;
      }

      // Escape — Close any open modal, or stop generation
      if (e.key === "Escape") {
        const settingsModal = dom.settingsModal;
        if (settingsModal && settingsModal.classList.contains("visible")) {
          e.preventDefault();
          closeSettings();
          return;
        }
        // Stop active streaming
        if (state.isStreaming && state.abortController) {
          e.preventDefault();
          state.abortController.abort();
          return;
        }
      }

      // Ctrl+, — Open settings
      if (mod && e.key === ",") {
        e.preventDefault();
        openSettings();
        return;
      }

      // Ctrl+Shift+S — Toggle sidebar
      if (mod && e.shiftKey && e.key === "S") {
        e.preventDefault();
        toggleSidebar();
        return;
      }
    });
  }

  // ---- Settings Modal ----

  // --- Memory Panel ---

  async function loadMemoryStats() {
    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(`${base}/v1/memory/stats`);
      if (!resp.ok) return;
      const data = await resp.json();
      const counts = data.counts || {};
      if (dom.memoryStatTotal) dom.memoryStatTotal.textContent = counts.total || 0;
      // Build detailed stats
      if (dom.memoryStats) {
        const types = ["fact", "preference", "entity", "narrative", "analysis"];
        let html = `<div class="memory-stat-row"><span class="memory-stat-label">Total:</span><span class="memory-stat-value">${counts.total || 0}</span></div>`;
        for (const t of types) {
          if (counts[t]) {
            html += `<div class="memory-stat-row"><span class="memory-stat-label">${t}:</span><span class="memory-stat-value">${counts[t]}</span></div>`;
          }
        }
        dom.memoryStats.innerHTML = html;
      }
    } catch (err) {
      console.debug("Failed to load memory stats:", err);
    }
  }

  async function loadMemoryList() {
    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(`${base}/v1/memory/facts?limit=50`);
      if (!resp.ok) return;
      const data = await resp.json();
      renderMemoryList(data.memories || []);
    } catch (err) {
      console.debug("Failed to load memories:", err);
    }
  }

  async function searchMemories() {
    const query = dom.memorySearchInput ? dom.memorySearchInput.value.trim() : "";
    if (!query) {
      loadMemoryList();
      return;
    }
    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(`${base}/v1/memory/search?q=${encodeURIComponent(query)}&limit=20`);
      if (!resp.ok) return;
      const data = await resp.json();
      renderMemoryList(data.results || []);
    } catch (err) {
      showToast("Memory search failed", "error");
    }
  }

  function renderMemoryList(memories) {
    if (!dom.memoryList) return;
    if (!memories.length) {
      dom.memoryList.innerHTML = "";
      return;
    }
    dom.memoryList.innerHTML = memories.map((m) => {
      // Tier is the subtractive-memory legibility surface: the user can
      // SEE what's earned a place (core/active) vs unproven (provisional)
      // vs tucked away (archive), and promote/demote it themselves.
      const tier = (m.tier || "active").toLowerCase();
      const tierTitle = {
        core: "Held close — always-in-context candidate",
        active: "Remembered — recalled when relevant",
        provisional: "Unproven — never injected; earns its place on re-mention",
        archive: "Archived — kept but out of the way",
      }[tier] || tier;
      const id = escapeHtml(m.id);
      // Keep (→core) is hidden when already core; Lower (→archive) hidden
      // when already archived — only show the moves that change something.
      const keepBtn = tier === "core" ? "" :
        `<button class="memory-tier-btn" data-act="keep" onclick="window._setMemoryTier('${id}', 'core')" title="Keep this close (promote to core)">&#x2605;</button>`;
      const lowerBtn = tier === "archive" ? "" :
        `<button class="memory-tier-btn" data-act="lower" onclick="window._setMemoryTier('${id}', 'archive')" title="Tuck away (demote to archive)">&#x2193;</button>`;
      // The Mirror: "why do I believe this?" reveals the evidence trail.
      return `
      <div class="memory-row" data-id="${id}">
        <div class="memory-item" data-tier="${escapeHtml(tier)}">
          <span class="memory-type-badge" data-type="${escapeHtml(m.memory_type)}">${escapeHtml(m.memory_type)}</span>
          <span class="memory-tier-badge" data-tier="${escapeHtml(tier)}" title="${escapeHtml(tierTitle)}">${escapeHtml(tier)}</span>
          <span class="memory-item-content" title="${escapeHtml(m.content)}">${escapeHtml(m.content)}</span>
          <button class="memory-why-btn" onclick="window._showBeliefEvidence('${id}')" title="Why do I believe this?">?</button>
          ${keepBtn}${lowerBtn}
          <button class="memory-delete-btn" onclick="window._deleteMemory('${id}')" title="Forget">&#x2715;</button>
        </div>
        <div class="memory-evidence" id="mem-ev-${id}" hidden></div>
      </div>`;
    }).join("");
  }

  window._showBeliefEvidence = async function (id) {
    const panel = document.getElementById(`mem-ev-${id}`);
    if (!panel) return;
    if (!panel.hidden) { panel.hidden = true; return; }  // toggle closed
    panel.hidden = false;
    panel.innerHTML = `<span class="memory-evidence-loading">Looking…</span>`;
    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(`${base}/v1/memory/facts/${id}/evidence`);
      if (!resp.ok) {
        panel.innerHTML = `<span class="memory-evidence-empty">Couldn't load the trail.</span>`;
        return;
      }
      const d = await resp.json();
      const conv = d.convergence || {};
      const n = conv.distinct_sources || 0;
      const convLine = n > 0
        ? `Conviction ${(conv.score || 0).toFixed(2)} — from ${n} independent source${n === 1 ? "" : "s"}`
        : `Not yet corroborated — earns its place when another signal agrees`;
      const days = d.days_since_reinforced;
      const stale = (typeof days === "number")
        ? `Last reinforced ${days === 0 ? "today" : days + " day" + (days === 1 ? "" : "s") + " ago"}`
        : "";
      const trail = (d.trail || []).map((t) =>
        `<li><span class="memory-ev-source">${escapeHtml(t.source)}</span> ${escapeHtml(t.claim || "")}</li>`
      ).join("");
      panel.innerHTML = `
        <div class="memory-evidence-origin">${escapeHtml(d.origin || "")}</div>
        <div class="memory-evidence-conv">${escapeHtml(convLine)}</div>
        ${trail ? `<ul class="memory-evidence-trail">${trail}</ul>` : ""}
        ${stale ? `<div class="memory-evidence-stale">${escapeHtml(stale)}</div>` : ""}`;
    } catch (err) {
      panel.innerHTML = `<span class="memory-evidence-empty">Couldn't load the trail.</span>`;
    }
  };

  window._setMemoryTier = async function (id, tier) {
    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(`${base}/v1/memory/facts/${id}/tier`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tier }),
      });
      if (resp.ok) {
        showToast(tier === "core" ? "Keeping that close" : "Moved to " + tier, "success");
        loadMemoryList();
        loadMemoryStats();
      } else {
        showToast("Couldn't change how that's kept", "error");
      }
    } catch (err) {
      showToast("Couldn't change how that's kept", "error");
    }
  };

  window._deleteMemory = async function (id) {
    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(`${base}/v1/memory/facts/${id}`, { method: "DELETE" });
      if (resp.ok) {
        showToast("Memory deleted", "success");
        loadMemoryList();
        loadMemoryStats();
      } else {
        showToast("Failed to delete memory", "error");
      }
    } catch (err) {
      showToast("Failed to delete memory", "error");
    }
  };

  async function addMemory() {
    const content = dom.memoryAddContent ? dom.memoryAddContent.value.trim() : "";
    const type = dom.memoryAddType ? dom.memoryAddType.value : "fact";
    if (!content) {
      showToast("Enter memory content first", "warning");
      return;
    }
    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(`${base}/v1/memory/store`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content, memory_type: type, importance: 0.8 }),
      });
      if (resp.ok) {
        showToast("Memory stored", "success");
        if (dom.memoryAddContent) dom.memoryAddContent.value = "";
        loadMemoryList();
        loadMemoryStats();
      } else {
        showToast("Failed to store memory", "error");
      }
    } catch (err) {
      showToast("Failed to store memory", "error");
    }
  }

  async function exportMemories() {
    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(`${base}/v1/memory/facts?limit=10000`);
      if (!resp.ok) return;
      const data = await resp.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `augmentum-memories-${new Date().toISOString().split("T")[0]}.json`;
      a.click();
      URL.revokeObjectURL(url);
      showToast("Memories exported", "success");
    } catch (err) {
      showToast("Export failed", "error");
    }
  }

  async function compactMemories() {
    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(`${base}/v1/memory/compact`, { method: "POST" });
      if (resp.ok) {
        showToast("Compaction complete", "success");
        loadMemoryStats();
      }
    } catch (err) {
      showToast("Compaction failed", "error");
    }
  }

  // ====== MCP Panel Functions ======

  async function mcpLoadServers() {
    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(`${base}/v1/mcp/servers`);
      const data = await resp.json();
      const container = dom.mcpServerList;
      if (!container) return;

      if (!data.enabled) {
        container.innerHTML = '<div class="mcp-empty">MCP is disabled</div>';
        return;
      }
      if (!data.servers || data.servers.length === 0) {
        container.innerHTML = '<div class="mcp-empty">No MCP servers connected</div>';
        return;
      }
      container.innerHTML = data.servers.map(s => `
        <div class="mcp-server-item">
          <div class="mcp-server-info">
            <span class="mcp-server-name">${escapeHtml(s.name)}</span>
            <span class="mcp-server-meta">${s.tool_count} tool${s.tool_count !== 1 ? 's' : ''}</span>
          </div>
          <button class="mcp-server-disconnect" onclick="window._mcpDisconnect('${escapeHtml(s.name)}')">Disconnect</button>
        </div>
      `).join("");
    } catch (err) {
      if (dom.mcpServerList) dom.mcpServerList.innerHTML = '<div class="mcp-empty">Failed to load servers</div>';
    }
  }

  async function mcpLoadTools() {
    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(`${base}/v1/mcp/tools`);
      const data = await resp.json();
      const container = dom.mcpToolList;
      if (!container) return;

      if (!data.tools || data.tools.length === 0) {
        container.innerHTML = '<div class="mcp-empty">No MCP tools available</div>';
        return;
      }
      container.innerHTML = data.tools.map(t => `
        <div class="mcp-tool-item">
          <span class="mcp-tool-name">${escapeHtml(t.name)}</span>
          <span class="mcp-tool-desc">${escapeHtml(t.description || '')}</span>
        </div>
      `).join("");
    } catch (err) {
      if (dom.mcpToolList) dom.mcpToolList.innerHTML = '<div class="mcp-empty">Failed to load tools</div>';
    }
  }

  async function mcpConnectServer() {
    const name = dom.mcpConnectName?.value?.trim();
    const type = dom.mcpConnectType?.value;
    const target = dom.mcpConnectTarget?.value?.trim();
    if (!name || !target) {
      showToast("Name and command/URL are required", "error");
      return;
    }

    const base = appSettings.backendUrl || "";
    const body = { name };
    if (type === "http") {
      body.url = target;
    } else {
      body.command = target;
      const argsStr = dom.mcpConnectArgs?.value?.trim();
      if (argsStr) body.args = argsStr.split(",").map(s => s.trim());
    }

    try {
      const resp = await fetch(`${base}/v1/mcp/connect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      if (resp.ok) {
        showToast(`Connected to ${name} (${data.tool_count} tools)`, "success");
        dom.mcpConnectName.value = "";
        dom.mcpConnectTarget.value = "";
        if (dom.mcpConnectArgs) dom.mcpConnectArgs.value = "";
        mcpLoadServers();
        mcpLoadTools();
      } else {
        showToast(data.error || "Connection failed", "error");
      }
    } catch (err) {
      showToast("Connection failed: " + err.message, "error");
    }
  }

  window._mcpDisconnect = async function(name) {
    const base = appSettings.backendUrl || "";
    try {
      const resp = await fetch(`${base}/v1/mcp/servers/${encodeURIComponent(name)}`, { method: "DELETE" });
      if (resp.ok) {
        showToast(`Disconnected from ${name}`, "success");
        mcpLoadServers();
        mcpLoadTools();
      } else {
        const data = await resp.json();
        showToast(data.error || "Disconnect failed", "error");
      }
    } catch (err) {
      showToast("Disconnect failed", "error");
    }
  };

  function openSettings() {
    // Reset to General tab
    dom.settingsTabs.querySelectorAll(".settings-tab").forEach((t) =>
      t.classList.toggle("active", t.dataset.tab === "general")
    );
    $$(".settings-tab-content").forEach((c) =>
      c.classList.toggle("active", c.id === "settings-tab-general")
    );

    // General tab
    dom.settingBackendUrl.value = appSettings.backendUrl;
    const modelOpt = dom.settingDefaultModel.querySelector(
      `option[value="${CSS.escape(appSettings.defaultModel)}"]`
    );
    if (modelOpt) {
      dom.settingDefaultModel.value = appSettings.defaultModel;
    }

    // HuggingFace token status
    dom.settingHfToken.value = "";
    dom.hfTokenStatus.className = "hf-token-status";
    dom.hfTokenStatus.title = "Checking...";
    fetch(`${appSettings.backendUrl || ""}/api/image/hf-token`)
      .then((r) => r.json())
      .then((data) => {
        if (data.is_set) {
          dom.hfTokenStatus.classList.add("set");
          dom.hfTokenStatus.title = "Token saved";
          dom.settingHfToken.placeholder = "Token saved (leave blank to keep)";
        } else {
          dom.hfTokenStatus.classList.add("unset");
          dom.hfTokenStatus.title = "No token set";
          dom.settingHfToken.placeholder = "hf_...";
        }
      })
      .catch(() => {
        dom.hfTokenStatus.classList.add("unset");
        dom.hfTokenStatus.title = "Could not check token status";
      });

    // Model tab
    dom.settingSystemPrompt.value = appSettings.systemPrompt || "";
    populateSliderPair(dom.settingTemperatureSlider, dom.settingTemperature, appSettings.temperature, 0.7);
    dom.settingMaxTokens.value = appSettings.maxTokens != null ? appSettings.maxTokens : "";
    dom.settingContextLimit.value = appSettings.contextLimit != null ? appSettings.contextLimit : "";
    populateSliderPair(dom.settingTopPSlider, dom.settingTopP, appSettings.topP, 1);

    // Advanced tab
    populateSliderPair(dom.settingFreqPenaltySlider, dom.settingFreqPenalty, appSettings.frequencyPenalty, 0);
    populateSliderPair(dom.settingPresPenaltySlider, dom.settingPresPenalty, appSettings.presencePenalty, 0);
    dom.settingSeed.value = appSettings.seed != null ? appSettings.seed : "";
    dom.settingStop.value = appSettings.stopSequences || "";

    // Tools tab — load from backend
    loadToolSettings();

    dom.settingsModal.classList.add("visible");
  }

  function populateSliderPair(slider, numberInput, value, sliderDefault) {
    if (value != null) {
      numberInput.value = value;
      slider.value = value;
    } else {
      numberInput.value = "";
      slider.value = sliderDefault;
    }
  }

  function closeSettings() {
    dom.settingsModal.classList.remove("visible");
  }

  function parseNumberOrNull(str) {
    if (str === "" || str == null) return null;
    const n = parseFloat(str);
    return isNaN(n) ? null : n;
  }

  function saveSettingsFromModal() {
    // General tab
    appSettings.backendUrl = dom.settingBackendUrl.value.trim();
    appSettings.defaultModel = dom.settingDefaultModel.value;
    // Theme is already applied via the toggle buttons

    // HuggingFace token — send to backend only, never store in localStorage
    const hfToken = dom.settingHfToken.value.trim();
    if (hfToken) {
      fetch(`${appSettings.backendUrl || ""}/api/image/hf-token`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: hfToken }),
      }).catch(() => {});
    }

    // Model tab
    appSettings.systemPrompt = dom.settingSystemPrompt.value.trim();
    appSettings.temperature = parseNumberOrNull(dom.settingTemperature.value);
    appSettings.maxTokens = parseNumberOrNull(dom.settingMaxTokens.value);
    appSettings.contextLimit = parseNumberOrNull(dom.settingContextLimit.value);
    appSettings.topP = parseNumberOrNull(dom.settingTopP.value);

    // Advanced tab
    appSettings.frequencyPenalty = parseNumberOrNull(dom.settingFreqPenalty.value);
    appSettings.presencePenalty = parseNumberOrNull(dom.settingPresPenalty.value);
    appSettings.seed = parseNumberOrNull(dom.settingSeed.value);
    appSettings.stopSequences = dom.settingStop.value.trim();

    saveSettings();

    // Tools tab — save to backend
    saveToolSettings();

    // Apply changes
    selectedModel = appSettings.defaultModel;
    dom.modelSelector.value = selectedModel;

    closeSettings();
    fetchModels(); // Re-fetch models with potentially new backend URL
  }

  // ---- Tool Settings (server-side) ----

  function loadToolSettings() {
    const base = appSettings.backendUrl || "";
    fetch(`${base}/api/config/tools`)
      .then((r) => r.json())
      .then((data) => {
        // Checkboxes
        $("#setting-auto-search").checked = data.uarf_auto_search !== false;
        $("#setting-auto-verify").checked = data.uarf_auto_verify !== false;
        $("#setting-proactive-search").checked = data.uarf_proactive_search !== false;
        $("#setting-proactive-math").checked = data.uarf_proactive_math !== false;
        $("#setting-proactive-code").checked = data.uarf_proactive_code !== false;
        $("#setting-heuristic-assess").checked = data.uarf_heuristic_assess !== false;

        // Sliders
        syncSliderPair("#setting-search-queries-slider", "#setting-search-queries", data.uarf_auto_search_queries ?? 5);
        syncSliderPair("#setting-search-results-slider", "#setting-search-results", data.uarf_auto_search_results_per_query ?? 4);
        syncSliderPair("#setting-search-context-slider", "#setting-search-context", data.uarf_auto_search_max_context_chars ?? 6000);
        syncSliderPair("#setting-search-retries-slider", "#setting-search-retries", data.uarf_search_retry_max ?? 1);
        syncSliderPair("#setting-max-tool-calls-slider", "#setting-max-tool-calls", data.uarf_max_tool_calls_per_phase ?? 3);

        // Sync condense model from server config
        var condenseSel = $("#img-condense-model");
        if (condenseSel && data.image_prompt_condense_model) {
          condenseSel.value = data.image_prompt_condense_model;
          appSettings.imgCondenseModel = data.image_prompt_condense_model;
        }
      })
      .catch(() => { /* backend unreachable — keep defaults */ });
  }

  function syncSliderPair(sliderSel, numberSel, value) {
    const slider = $(sliderSel);
    const number = $(numberSel);
    if (slider) slider.value = value;
    if (number) number.value = value;
  }

  function saveToolSettings() {
    const base = appSettings.backendUrl || "";
    const body = {
      uarf_auto_search: $("#setting-auto-search").checked,
      uarf_auto_search_queries: parseInt($("#setting-search-queries").value) || 5,
      uarf_auto_search_results_per_query: parseInt($("#setting-search-results").value) || 4,
      uarf_auto_search_max_context_chars: parseInt($("#setting-search-context").value) || 6000,
      uarf_search_retry_max: parseInt($("#setting-search-retries").value) || 1,
      uarf_auto_verify: $("#setting-auto-verify").checked,
      uarf_proactive_search: $("#setting-proactive-search").checked,
      uarf_proactive_math: $("#setting-proactive-math").checked,
      uarf_proactive_code: $("#setting-proactive-code").checked,
      uarf_heuristic_assess: $("#setting-heuristic-assess").checked,
      uarf_max_tool_calls_per_phase: parseInt($("#setting-max-tool-calls").value) || 3,
    };
    fetch(`${base}/api/config/tools`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).catch(() => { /* silent */ });
  }

  // ---- Model Manager ----

  function openModelManager() {
    dom.modelManagerModal.classList.add("visible");
    dom.mmPullInput.value = "";
    hideProgress();
    refreshModelList();
  }

  function closeModelManager() {
    dom.modelManagerModal.classList.remove("visible");
    // Cancel any in-progress download
    if (pullAbortController) {
      pullAbortController.abort();
      pullAbortController = null;
    }
  }

  async function refreshModelList() {
    const list = dom.mmModelList;
    list.innerHTML = '<div class="mm-empty">Loading models...</div>';

    try {
      const base = appSettings.backendUrl || "";
      // Fetch models and running status in parallel
      const [tagsResp, runningResp] = await Promise.all([
        fetch(`${base}/api/tags`),
        fetch(`${base}/api/ps`).catch(() => null),
      ]);

      if (!tagsResp.ok) throw new Error("Failed to fetch models");

      const tagsData = await tagsResp.json();
      const models = (tagsData.models || []).filter(
        (m) => !m.name.startsWith("a/") && !m.name.startsWith("n/") && !m.name.startsWith("p/")
      );

      let runningModels = [];
      if (runningResp && runningResp.ok) {
        const runningData = await runningResp.json();
        runningModels = (runningData.models || []).map((m) => m.name);
      }

      if (models.length === 0) {
        list.innerHTML = `
          <div class="mm-empty-welcome">
            <p>No models downloaded yet. Enter a model name above or click a suggestion to get started.</p>
            <div class="mm-chips">
              <button class="mm-chip" data-model="llama3.1:8b">llama3.1:8b</button>
              <button class="mm-chip" data-model="mistral">mistral</button>
              <button class="mm-chip" data-model="gemma2">gemma2</button>
            </div>
          </div>
        `;
        list.querySelectorAll(".mm-chip").forEach((btn) => {
          btn.addEventListener("click", () => pullModel(btn.dataset.model));
        });
        return;
      }

      list.innerHTML = "";
      for (const model of models) {
        const backend = (model.details && model.details.augmentum_backend) || "ollama";
        const isOllama = backend === "ollama";
        const isLoaded = isOllama && runningModels.some(
          (r) => r === model.name || r.startsWith(model.name.split(":")[0])
        );
        const card = createModelCard(model, isLoaded, isOllama);
        list.appendChild(card);
      }
    } catch (err) {
      list.innerHTML = `<div class="mm-empty">Could not load models. Is the backend running?</div>`;
    }
  }

  function createModelCard(model, isLoaded, isOllama) {
    const card = document.createElement("div");
    card.className = "mm-model-card";
    card.dataset.name = model.name;

    const sizeStr = model.size ? formatBytes(model.size) : "";
    const details = model.details || {};
    const backend = details.augmentum_backend || "ollama";
    const metaParts = [];
    if (details.quantization_level) metaParts.push(details.quantization_level);
    if (details.parameter_size) metaParts.push(details.parameter_size);
    else if (details.family) metaParts.push(details.family);
    if (sizeStr) metaParts.push(sizeStr);
    if (!isOllama) metaParts.push(backend);

    const statusDot = isOllama
      ? `<div class="mm-status-dot ${isLoaded ? "loaded" : "unloaded"}" title="${isLoaded ? "Loaded in memory" : "Not loaded"}"></div>`
      : `<div class="mm-status-dot loaded" title="Managed by ${escapeHtml(backend)}"></div>`;

    let actionsHtml;
    if (isOllama) {
      actionsHtml = `
        ${isLoaded
          ? '<button class="btn btn-secondary btn-sm mm-unload-btn">Unload</button>'
          : '<button class="btn btn-secondary btn-sm mm-load-btn">Load</button>'
        }
        <button class="btn btn-danger btn-sm mm-delete-btn">Delete</button>
      `;
    } else {
      actionsHtml = `<span class="prov-builtin-badge">${escapeHtml(backend)}</span>`;
    }

    card.innerHTML = `
      ${statusDot}
      <div class="mm-model-info">
        <div class="mm-model-name">${escapeHtml(model.name)}</div>
        <div class="mm-model-meta">${escapeHtml(metaParts.join(" \u00b7 "))}</div>
      </div>
      <div class="mm-model-actions">
        ${actionsHtml}
      </div>
    `;

    if (isOllama) {
      // Load / Unload
      const loadBtn = card.querySelector(".mm-load-btn");
      const unloadBtn = card.querySelector(".mm-unload-btn");
      if (loadBtn) {
        loadBtn.addEventListener("click", () => loadModel(model.name));
      }
      if (unloadBtn) {
        unloadBtn.addEventListener("click", () => unloadModel(model.name));
      }

      // Delete
      card.querySelector(".mm-delete-btn").addEventListener("click", () => {
        showDeleteConfirm(model.name, card);
      });
    }

    return card;
  }

  function showDeleteConfirm(modelName, cardEl) {
    const actions = cardEl.querySelector(".mm-model-actions");
    const originalHtml = actions.innerHTML;

    actions.innerHTML = `
      <div class="mm-delete-confirm">
        <span>Delete ${escapeHtml(modelName)}?</span>
        <button class="btn btn-danger btn-sm mm-confirm-yes">Yes, delete</button>
        <button class="btn btn-secondary btn-sm mm-confirm-no">Cancel</button>
      </div>
    `;

    actions.querySelector(".mm-confirm-yes").addEventListener("click", async () => {
      await deleteModel(modelName);
    });

    actions.querySelector(".mm-confirm-no").addEventListener("click", () => {
      actions.innerHTML = originalHtml;
      // Re-bind events
      const newCard = createModelCard(
        { name: modelName, size: null, details: {} },
        false
      );
      // Simpler: just refresh the whole list
      refreshModelList();
    });
  }

  async function pullModel(name) {
    if (!name || !name.trim()) return;
    name = name.trim();

    // Show progress
    dom.mmProgressArea.classList.remove("hidden");
    dom.mmProgressModel.textContent = name;
    dom.mmProgressFill.style.width = "0%";
    dom.mmProgressStatus.textContent = "Preparing...";
    dom.mmPullBtn.disabled = true;

    pullAbortController = new AbortController();

    try {
      const base = appSettings.backendUrl || "";
      const response = await fetch(`${base}/api/pull`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name, stream: true }),
        signal: pullAbortController.signal,
      });

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(errBody.detail || errBody.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            updatePullProgress(data);
          } catch {
            // skip malformed
          }
        }
      }

      // Process remaining buffer
      if (buffer.trim()) {
        try {
          const data = JSON.parse(buffer);
          updatePullProgress(data);
        } catch {
          // skip
        }
      }

      // Success
      dom.mmProgressFill.style.width = "100%";
      dom.mmProgressStatus.textContent = "Download complete!";
      setTimeout(() => {
        hideProgress();
        refreshModelList();
        fetchModels(); // Refresh the main model selector
      }, 1500);
    } catch (err) {
      if (err.name === "AbortError") {
        dom.mmProgressStatus.textContent = "Download cancelled.";
        setTimeout(hideProgress, 1000);
      } else {
        dom.mmProgressStatus.textContent = "Error: " + err.message;
        dom.mmProgressFill.style.width = "0%";
      }
    }

    pullAbortController = null;
    dom.mmPullBtn.disabled = false;
  }

  function updatePullProgress(data) {
    const status = data.status || "";
    dom.mmProgressStatus.textContent = status;

    if (data.total && data.completed !== undefined) {
      const pct = Math.round((data.completed / data.total) * 100);
      dom.mmProgressFill.style.width = pct + "%";

      const completedStr = formatBytes(data.completed);
      const totalStr = formatBytes(data.total);
      dom.mmProgressStatus.textContent = `${status} \u2014 ${completedStr} / ${totalStr} (${pct}%)`;
    } else if (status === "success") {
      dom.mmProgressFill.style.width = "100%";
    }
  }

  function hideProgress() {
    dom.mmProgressArea.classList.add("hidden");
    dom.mmProgressFill.style.width = "0%";
    dom.mmProgressStatus.textContent = "";
  }

  // --- Backend toggle & GGUF support ---

  const ollamaChips = [
    { label: "qwen3:8b", model: "qwen3:8b" },
    { label: "llama3.3", model: "llama3.3" },
    { label: "gemma3:12b", model: "gemma3:12b" },
    { label: "phi4", model: "phi4" },
    { label: "mistral-small3.2", model: "mistral-small3.2" },
    { label: "deepseek-r1:8b", model: "deepseek-r1:8b" },
  ];

  const llamacppChips = [
    { label: "Qwen 3.6 27B", model: "unsloth/Qwen3.6-27B-GGUF" },
    { label: "Qwen 3.6 35B A3B", model: "unsloth/Qwen3.6-35B-A3B-GGUF" },
    { label: "Gemma 4 E4B", model: "unsloth/gemma-4-E4B-it-GGUF" },
    { label: "Phi 4", model: "unsloth/phi-4-GGUF" },
    { label: "GLM 4.7 Flash", model: "unsloth/GLM-4.7-Flash-GGUF" },
  ];

  function updateMmBackendUI() {
    const backend = dom.mmBackendSelect.value;
    const isLlamacpp = backend === "llamacpp";

    // Update placeholder
    dom.mmPullInput.placeholder = isLlamacpp
      ? "e.g. unsloth/Qwen3.6-27B-GGUF"
      : "e.g. llama3.1:8b";

    // Update button text
    dom.mmPullBtn.textContent = isLlamacpp ? "List Files" : "Download";

    // Update chips
    const chips = isLlamacpp ? llamacppChips : ollamaChips;
    dom.mmChips.innerHTML = chips
      .map((c) => `<button class="mm-chip" data-model="${c.model}">${c.label}</button>`)
      .join("");
    dom.mmChips.querySelectorAll(".mm-chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (isLlamacpp) {
          dom.mmPullInput.value = btn.dataset.model;
          fetchGgufFiles(btn.dataset.model);
        } else {
          pullModel(btn.dataset.model);
        }
      });
    });

    // Update browse link
    if (isLlamacpp) {
      dom.mmBrowseLink.innerHTML =
        '<a href="https://huggingface.co/models?sort=trending&search=gguf" target="_blank" rel="noopener">Browse GGUF models on HuggingFace <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg></a>';
    } else {
      dom.mmBrowseLink.innerHTML =
        '<a href="https://ollama.com/search" target="_blank" rel="noopener">Browse models on ollama.com <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg></a>';
    }

    // Hide GGUF picker when switching
    dom.mmGgufPicker.classList.add("hidden");
    dom.mmGgufList.innerHTML = "";
  }

  async function fetchGgufFiles(repoId) {
    if (!repoId) return;

    dom.mmGgufPicker.classList.remove("hidden");
    dom.mmGgufList.innerHTML = '<div class="mm-empty">Loading files\u2026</div>';

    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(`${base}/api/models/gguf/list?repo=${encodeURIComponent(repoId)}`);
      const data = await resp.json();

      if (data.error) {
        dom.mmGgufList.innerHTML = `<div class="mm-empty">${data.error}</div>`;
        return;
      }

      if (!data.files || data.files.length === 0) {
        dom.mmGgufList.innerHTML = '<div class="mm-empty">No .gguf files found in this repo.</div>';
        return;
      }

      dom.mmGgufList.innerHTML = data.files
        .map((f) => {
          const sizeStr = formatBytes(f.size);
          return `<div class="mm-gguf-item" data-filename="${f.filename}" data-repo="${repoId}">
            <span class="mm-gguf-name">${f.filename}</span>
            <span class="mm-gguf-size">${sizeStr}</span>
            <button class="btn btn-primary mm-gguf-dl-btn">Download</button>
          </div>`;
        })
        .join("");

      // Bind download buttons
      dom.mmGgufList.querySelectorAll(".mm-gguf-dl-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          const item = btn.closest(".mm-gguf-item");
          pullGgufModel(item.dataset.repo, item.dataset.filename);
        });
      });
    } catch (err) {
      dom.mmGgufList.innerHTML = `<div class="mm-empty">Error: ${err.message}</div>`;
    }
  }

  async function pullGgufModel(repoId, filename) {
    // Show progress
    dom.mmProgressArea.classList.remove("hidden");
    dom.mmProgressModel.textContent = filename;
    dom.mmProgressFill.style.width = "0%";
    dom.mmProgressStatus.textContent = "Preparing...";
    dom.mmPullBtn.disabled = true;

    pullAbortController = new AbortController();

    try {
      const base = appSettings.backendUrl || "";
      const response = await fetch(`${base}/api/models/pull`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ backend: "llamacpp", name: repoId, filename }),
        signal: pullAbortController.signal,
      });

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(errBody.detail || errBody.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            updateGgufProgress(data);
          } catch {
            // skip malformed
          }
        }
      }

      if (buffer.trim()) {
        try {
          const data = JSON.parse(buffer);
          updateGgufProgress(data);
        } catch {
          // skip
        }
      }

      dom.mmProgressFill.style.width = "100%";
      dom.mmProgressStatus.textContent = "Download complete!";
      setTimeout(() => {
        hideProgress();
        refreshModelList();
        fetchModels();
      }, 1500);
    } catch (err) {
      if (err.name === "AbortError") {
        dom.mmProgressStatus.textContent = "Download cancelled.";
        setTimeout(hideProgress, 1000);
      } else {
        dom.mmProgressStatus.textContent = "Error: " + err.message;
        dom.mmProgressFill.style.width = "0%";
      }
    }

    pullAbortController = null;
    dom.mmPullBtn.disabled = false;
  }

  function updateGgufProgress(data) {
    const status = data.status || "";
    if (status === "downloading") {
      const totalStr = data.total ? formatBytes(data.total) : "";
      dom.mmProgressStatus.textContent = totalStr
        ? `Downloading... (${totalStr})`
        : "Downloading...";
    } else if (status === "complete") {
      const sizeStr = data.size ? formatBytes(data.size) : "";
      dom.mmProgressStatus.textContent = `Complete${sizeStr ? " — " + sizeStr : ""}`;
      dom.mmProgressFill.style.width = "100%";
    } else if (status === "exists") {
      dom.mmProgressStatus.textContent = "File already exists.";
      dom.mmProgressFill.style.width = "100%";
    } else if (status === "error") {
      dom.mmProgressStatus.textContent = "Error: " + (data.error || "Unknown error");
      dom.mmProgressFill.style.width = "0%";
    } else {
      dom.mmProgressStatus.textContent = status;
    }
  }

  async function deleteModel(name) {
    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(`${base}/api/delete`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name }),
      });

      if (!resp.ok) throw new Error("Delete failed");

      refreshModelList();
      fetchModels(); // Refresh main model selector
    } catch (err) {
      showToast("Failed to delete model: " + err.message, "error");
    }
  }

  async function loadModel(name) {
    try {
      const base = appSettings.backendUrl || "";
      // Encode model name for URL (handles colons, slashes)
      const encodedName = encodeURIComponent(name);
      const resp = await fetch(`${base}/api/models/${encodedName}/load`, {
        method: "POST",
      });
      if (!resp.ok) throw new Error("Load failed");
      refreshModelList();
    } catch (err) {
      showToast("Failed to load model: " + err.message, "error");
    }
  }

  async function unloadModel(name) {
    try {
      const base = appSettings.backendUrl || "";
      const encodedName = encodeURIComponent(name);
      const resp = await fetch(`${base}/api/models/${encodedName}/unload`, {
        method: "POST",
      });
      if (!resp.ok) throw new Error("Unload failed");
      refreshModelList();
    } catch (err) {
      showToast("Failed to unload model: " + err.message, "error");
    }
  }

  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return "0 B";
    // Use decimal (SI) units to match HuggingFace / catalog sizes
    const units = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(1000));
    const val = bytes / Math.pow(1000, i);
    return val.toFixed(i > 1 ? 1 : 0) + " " + units[i];
  }

  // ---- Provider Manager ----

  function openProviderModal() {
    dom.provName.value = "";
    dom.provUrl.value = "";
    dom.provKey.value = "";
    dom.provTestResult.classList.add("hidden");
    dom.providerModal.classList.add("visible");
    refreshProviderList();
  }

  function closeProviderModal() {
    dom.providerModal.classList.remove("visible");
  }

  async function testProviderConnection() {
    const url = dom.provUrl.value.trim();
    if (!url) {
      showProvTestResult("error", "Please enter a base URL.");
      return;
    }

    dom.provTestBtn.disabled = true;
    dom.provTestBtn.textContent = "Testing...";

    try {
      const base = appSettings.backendUrl || "";
      const body = { base_url: url };
      const key = dom.provKey.value.trim();
      if (key) body.api_key = key;

      const resp = await fetch(`${base}/api/providers/probe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await resp.json();

      if (data.status === "ok" && data.models) {
        const names = data.models.map((m) => m.name).slice(0, 10);
        const extra = data.models.length > 10 ? ` (+${data.models.length - 10} more)` : "";
        showProvTestResult(
          "success",
          `Connected! Found ${data.models.length} model(s): ${names.join(", ")}${extra}`
        );
      } else {
        showProvTestResult("error", data.error || "Connection failed.");
      }
    } catch (err) {
      showProvTestResult("error", "Connection failed: " + err.message);
    } finally {
      dom.provTestBtn.disabled = false;
      dom.provTestBtn.textContent = "Test Connection";
    }
  }

  function showProvTestResult(type, message) {
    const el = dom.provTestResult;
    el.className = "prov-test-result " + type;
    el.textContent = message;
  }

  async function addProvider() {
    const name = dom.provName.value.trim();
    const url = dom.provUrl.value.trim();
    if (!name || !url) {
      showProvTestResult("error", "Name and URL are required.");
      return;
    }

    // Generate slug ID from name
    const id = name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
    if (!id) {
      showProvTestResult("error", "Invalid name — cannot generate ID.");
      return;
    }

    const body = { id, name, base_url: url };
    const key = dom.provKey.value.trim();
    if (key) body.api_key = key;

    dom.provAddBtn.disabled = true;

    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(`${base}/api/providers`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await resp.json();

      if (resp.ok) {
        dom.provName.value = "";
        dom.provUrl.value = "";
        dom.provKey.value = "";
        dom.provTestResult.classList.add("hidden");
        refreshProviderList();
        fetchModels(); // Refresh model selector
        fetchCapabilities(); // Update UI for new backend
      } else {
        showProvTestResult("error", data.error || "Failed to add provider.");
      }
    } catch (err) {
      showProvTestResult("error", "Failed: " + err.message);
    } finally {
      dom.provAddBtn.disabled = false;
    }
  }

  async function refreshProviderList() {
    const list = dom.provList;
    list.innerHTML = '<div class="mm-empty">Loading...</div>';

    try {
      const base = appSettings.backendUrl || "";
      const resp = await fetch(`${base}/api/providers`);
      if (!resp.ok) throw new Error("fetch failed");
      const data = await resp.json();
      const providers = data.providers || [];

      if (providers.length === 0) {
        list.innerHTML = '<div class="mm-empty">No providers configured.</div>';
        return;
      }

      list.innerHTML = "";
      providers.forEach((p) => list.appendChild(createProviderCard(p)));
    } catch {
      list.innerHTML = '<div class="mm-empty">Failed to load providers.</div>';
    }
  }

  function createProviderCard(provider) {
    const card = document.createElement("div");
    card.className = "prov-card";

    const info = document.createElement("div");
    info.className = "prov-card-info";

    const nameEl = document.createElement("div");
    nameEl.className = "prov-card-name";
    nameEl.textContent = provider.name;
    info.appendChild(nameEl);

    if (provider.type === "builtin") {
      const badge = document.createElement("span");
      badge.className = "prov-builtin-badge";
      badge.textContent = "Built-in";
      info.appendChild(badge);
    } else {
      const urlEl = document.createElement("div");
      urlEl.className = "prov-card-url";
      urlEl.textContent = provider.base_url || "";
      info.appendChild(urlEl);
    }

    card.appendChild(info);

    if (provider.type === "user") {
      const actions = document.createElement("div");
      actions.className = "prov-card-actions";

      const testBtn = document.createElement("button");
      testBtn.className = "btn btn-secondary btn-sm";
      testBtn.textContent = "Test";
      testBtn.addEventListener("click", async () => {
        testBtn.disabled = true;
        testBtn.textContent = "...";
        try {
          const base = appSettings.backendUrl || "";
          const resp = await fetch(`${base}/api/providers/${provider.id}/test`, {
            method: "POST",
          });
          const data = await resp.json();
          testBtn.textContent = data.status === "ok" ? "OK" : "Fail";
        } catch {
          testBtn.textContent = "Fail";
        }
        setTimeout(() => {
          testBtn.textContent = "Test";
          testBtn.disabled = false;
        }, 2000);
      });
      actions.appendChild(testBtn);

      const delBtn = document.createElement("button");
      delBtn.className = "btn btn-danger btn-sm";
      delBtn.textContent = "Remove";
      delBtn.addEventListener("click", () => deleteProvider(provider.id));
      actions.appendChild(delBtn);

      card.appendChild(actions);
    }

    return card;
  }

  async function deleteProvider(id) {
    if (!confirm("Remove this provider? Its models will no longer be available.")) return;

    try {
      const base = appSettings.backendUrl || "";
      await fetch(`${base}/api/providers/${id}`, { method: "DELETE" });
      refreshProviderList();
      fetchModels();
      fetchCapabilities();
    } catch {
      // silently fail
    }
  }

  // ==========================================================================
  // Image Generation Panel
  // ==========================================================================

  let imageGenerating = false;
  let imageAbortController = null; // AbortController for cancelling generation
  let imageHardwareInfo = null; // stored hardware profile from backend
  let imageModelsData = [];     // cached model list from /api/image/models

  // Resolution presets per pipeline type
  var IMG_RESOLUTION_PRESETS = {
    sd15: [
      { label: "512\u00d7512", w: 512, h: 512 },
      { label: "512\u00d7768", w: 512, h: 768 },
      { label: "768\u00d7512", w: 768, h: 512 },
      { label: "768\u00d7768", w: 768, h: 768 },
    ],
    sdxl: [
      { label: "1024\u00d71024", w: 1024, h: 1024 },
      { label: "832\u00d71216", w: 832, h: 1216 },
      { label: "1216\u00d7832", w: 1216, h: 832 },
      { label: "768\u00d71344", w: 768, h: 1344 },
      { label: "1344\u00d7768", w: 1344, h: 768 },
    ],
    flux: [
      { label: "1024\u00d71024", w: 1024, h: 1024 },
      { label: "832\u00d71216", w: 832, h: 1216 },
      { label: "1216\u00d7832", w: 1216, h: 832 },
      { label: "768\u00d71344", w: 768, h: 1344 },
      { label: "1344\u00d7768", w: 1344, h: 768 },
    ],
  };

  function getSelectedPipelineType() {
    var modelEl = $("#img-model");
    var modelName = modelEl ? modelEl.value : "";
    if (modelName && imageModelsData.length) {
      var found = imageModelsData.find(function (m) { return m.name === modelName; });
      if (found) return found.pipeline_type;
    }
    // Fallback: use hardware recommended pipeline or sd15
    if (imageHardwareInfo && imageHardwareInfo.recommended_pipeline) {
      return imageHardwareInfo.recommended_pipeline;
    }
    return "sd15";
  }

  function renderResolutionPresets() {
    var container = $("#img-resolution-presets");
    if (!container) return;
    var ptype = getSelectedPipelineType();
    var presets = IMG_RESOLUTION_PRESETS[ptype] || IMG_RESOLUTION_PRESETS.sd15;
    var widthEl = $("#img-width");
    var heightEl = $("#img-height");
    var curW = widthEl ? parseInt(widthEl.value) || 0 : 0;
    var curH = heightEl ? parseInt(heightEl.value) || 0 : 0;

    container.innerHTML = "";
    presets.forEach(function (p) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "img-resolution-preset";
      btn.textContent = p.label;
      if (p.w === curW && p.h === curH) btn.classList.add("active");
      btn.addEventListener("click", function () {
        if (widthEl) widthEl.value = p.w;
        if (heightEl) heightEl.value = p.h;
        container.querySelectorAll(".img-resolution-preset").forEach(function (b) { b.classList.remove("active"); });
        btn.classList.add("active");
        saveImageSettings();
      });
      container.appendChild(btn);
    });
  }

  async function fetchImageSamplers() {
    try {
      var base = appSettings.backendUrl || "";
      var resp = await fetch(base + "/api/image/samplers");
      if (!resp.ok) return;
      var samplers = await resp.json();
      var select = $("#img-sampler");
      if (!select) return;
      select.innerHTML = '<option value="">Default</option>';
      samplers.forEach(function (s) {
        var opt = document.createElement("option");
        opt.value = s.name;
        opt.textContent = s.display_name;
        select.appendChild(opt);
      });
      // Restore persisted sampler selection
      if (appSettings.imgSampler) {
        select.value = appSettings.imgSampler;
      }
    } catch (_e) { /* samplers endpoint not available */ }
  }

  async function fetchImageHardware() {
    var base = appSettings.backendUrl || "";
    var container = $("#img-hardware-info");
    if (!container) return;

    try {
      var resp = await fetch(base + "/api/image/hardware");
      if (resp.status === 503) {
        // Image generation not enabled — keep hidden
        return;
      }
      if (!resp.ok) return;

      var hw = await resp.json();
      imageHardwareInfo = hw;

      // Device name
      var deviceEl = $("#img-hw-device");
      if (deviceEl) deviceEl.textContent = hw.device_name || hw.device || "Unknown";

      // Tier badge
      var tierEl = $("#img-hw-tier");
      if (tierEl) {
        var tier = (hw.tier || "cpu").toLowerCase();
        tierEl.textContent = tier.toUpperCase();
        tierEl.className = "img-hw-tier-badge tier-" + tier;
      }

      // VRAM bar
      var vramRow = $("#img-hw-vram-row");
      var vramFill = $("#img-hw-vram-fill");
      var vramText = $("#img-hw-vram-text");
      if (hw.vram_total_mb > 0) {
        var usedMb = hw.vram_total_mb - hw.vram_free_mb;
        var pct = Math.min(100, Math.round((usedMb / hw.vram_total_mb) * 100));
        if (vramFill) vramFill.style.width = pct + "%";
        if (vramText) vramText.textContent =
          (usedMb / 1024).toFixed(1) + " / " + (hw.vram_total_mb / 1024).toFixed(1) + " GB";
        if (vramRow) vramRow.style.display = "";
      } else {
        if (vramRow) vramRow.style.display = "none";
      }

      // Recommendation
      var recEl = $("#img-hw-rec");
      if (recEl && hw.recommended_pipeline) {
        recEl.textContent = "Recommended: " + hw.recommended_pipeline.toUpperCase();
      }

      container.classList.remove("hidden");
      renderResolutionPresets();
    } catch (_e) {
      // silently ignore — hardware info is optional
    }
  }

  function saveImageSettings() {
    var w = $("#img-width"), h = $("#img-height"), s = $("#img-steps"), c = $("#img-cfg");
    var sd = $("#img-seed"), sm = $("#img-sampler"), m = $("#img-model"), p = $("#img-preset");
    var n = $("#img-negative");
    appSettings.imgWidth = w ? parseInt(w.value) || null : null;
    appSettings.imgHeight = h ? parseInt(h.value) || null : null;
    appSettings.imgSteps = s ? parseInt(s.value) || null : null;
    appSettings.imgCfg = c ? parseFloat(c.value) || null : null;
    appSettings.imgSeed = sd ? parseInt(sd.value) : null;
    appSettings.imgSampler = sm ? sm.value : "";
    appSettings.imgModel = m ? m.value : "";
    appSettings.imgPreset = p ? p.value : "";
    appSettings.imgNegative = n ? n.value : "";
    var cm = $("#img-condense-model");
    appSettings.imgCondenseModel = cm ? cm.value : "";
    saveSettings();
  }

  function restoreImageSettings() {
    var s = appSettings;
    if (s.imgWidth) { var el = $("#img-width"); if (el) el.value = s.imgWidth; }
    if (s.imgHeight) { var el = $("#img-height"); if (el) el.value = s.imgHeight; }
    if (s.imgSteps) { var el = $("#img-steps"); if (el) el.value = s.imgSteps; }
    if (s.imgCfg != null) { var el = $("#img-cfg"); if (el) el.value = s.imgCfg; }
    if (s.imgSeed != null) { var el = $("#img-seed"); if (el) el.value = s.imgSeed; }
    if (s.imgSampler) { var el = $("#img-sampler"); if (el) el.value = s.imgSampler; }
    if (s.imgPreset) { var el = $("#img-preset"); if (el) el.value = s.imgPreset; }
    if (s.imgNegative) { var el = $("#img-negative"); if (el) el.value = s.imgNegative; }
    // imgModel is restored after model list loads (in refreshImageModels)
  }

  // --- Image editing mode state ---
  var currentImageMode = "txt2img";
  var sourceImageBase64 = "";   // raw base64 (no data URI prefix)
  var maskCanvasCtx = null;     // offscreen mask context (clean B/W)
  var maskOverlayCtx = null;    // on-screen overlay context (red visualization)
  var maskBrushSize = 30;
  var maskTool = "brush";       // "brush" or "eraser"
  var maskPainting = false;

  function setImageMode(mode) {
    currentImageMode = mode;
    // Update tabs
    document.querySelectorAll(".img-mode-tab").forEach(function (t) {
      t.classList.toggle("active", t.dataset.mode === mode);
    });
    // Show/hide source section
    var srcSection = $("#img-source-section");
    var maskSection = $("#img-mask-section");
    if (srcSection) srcSection.classList.toggle("hidden", mode === "txt2img");
    if (maskSection) maskSection.classList.toggle("hidden", mode !== "inpaint");

    // Update generate button text
    var genBtn = $("#img-generate-btn");
    if (genBtn) {
      if (mode === "txt2img") genBtn.textContent = "Generate";
      else if (mode === "img2img") genBtn.textContent = "Transform";
      else genBtn.textContent = "Inpaint";
    }

    // Default strength
    var strengthEl = $("#img-strength");
    var strengthValEl = $("#img-strength-value");
    if (strengthEl) {
      if (mode === "inpaint") { strengthEl.value = "1.0"; }
      else { strengthEl.value = "0.75"; }
      if (strengthValEl) strengthValEl.textContent = strengthEl.value;
    }
  }

  function loadSourceImage(file) {
    var reader = new FileReader();
    reader.onload = function (e) {
      var dataUrl = e.target.result;
      sourceImageBase64 = dataUrl.split(",")[1];
      var preview = $("#img-source-preview");
      var placeholder = $("#img-source-placeholder");
      if (preview) {
        preview.src = dataUrl;
        preview.classList.remove("hidden");
      }
      if (placeholder) placeholder.classList.add("hidden");

      // If inpaint mode, set up mask canvas
      if (currentImageMode === "inpaint") {
        setupMaskCanvas(dataUrl);
      }
    };
    reader.readAsDataURL(file);
  }

  function loadSourceImageFromUrl(url) {
    var img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = function () {
      var canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      var ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      var dataUrl = canvas.toDataURL("image/png");
      sourceImageBase64 = dataUrl.split(",")[1];
      var preview = $("#img-source-preview");
      var placeholder = $("#img-source-placeholder");
      if (preview) {
        preview.src = dataUrl;
        preview.classList.remove("hidden");
      }
      if (placeholder) placeholder.classList.add("hidden");
      if (currentImageMode === "inpaint") {
        setupMaskCanvas(dataUrl);
      }
    };
    img.src = url;
  }

  function setupMaskCanvas(imageDataUrl) {
    var canvas = $("#img-mask-canvas");
    var wrap = $("#img-mask-canvas-wrap");
    if (!canvas || !wrap) return;

    var img = new Image();
    img.onload = function () {
      // Size canvas to image (max 512px display width)
      var displayW = Math.min(img.naturalWidth, 512);
      var scale = displayW / img.naturalWidth;
      var displayH = Math.round(img.naturalHeight * scale);

      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.style.width = displayW + "px";
      canvas.style.height = displayH + "px";

      // Draw source image as background
      maskOverlayCtx = canvas.getContext("2d");
      maskOverlayCtx.drawImage(img, 0, 0);

      // Create offscreen canvas for clean mask data
      var offscreen = document.createElement("canvas");
      offscreen.width = img.naturalWidth;
      offscreen.height = img.naturalHeight;
      maskCanvasCtx = offscreen.getContext("2d");
      maskCanvasCtx.fillStyle = "black";
      maskCanvasCtx.fillRect(0, 0, offscreen.width, offscreen.height);

      // Store image ref for redraw
      canvas._sourceImg = img;
    };
    img.src = imageDataUrl;
  }

  function maskCanvasPointerDown(e) {
    maskPainting = true;
    maskCanvasPaint(e);
  }

  function maskCanvasPointerMove(e) {
    if (!maskPainting) return;
    maskCanvasPaint(e);
  }

  function maskCanvasPointerUp() {
    maskPainting = false;
  }

  function maskCanvasPaint(e) {
    var canvas = $("#img-mask-canvas");
    if (!canvas || !maskCanvasCtx || !maskOverlayCtx) return;

    var rect = canvas.getBoundingClientRect();
    var scaleX = canvas.width / rect.width;
    var scaleY = canvas.height / rect.height;
    var x = (e.clientX - rect.left) * scaleX;
    var y = (e.clientY - rect.top) * scaleY;
    var radius = maskBrushSize * scaleX;

    // Paint on offscreen mask
    maskCanvasCtx.beginPath();
    maskCanvasCtx.arc(x, y, radius, 0, Math.PI * 2);
    if (maskTool === "brush") {
      maskCanvasCtx.fillStyle = "white";
    } else {
      maskCanvasCtx.fillStyle = "black";
    }
    maskCanvasCtx.fill();

    // Redraw overlay: source image + semi-transparent red where mask is white
    maskOverlayCtx.clearRect(0, 0, canvas.width, canvas.height);
    if (canvas._sourceImg) {
      maskOverlayCtx.drawImage(canvas._sourceImg, 0, 0);
    }
    // Overlay mask in red
    maskOverlayCtx.save();
    maskOverlayCtx.globalAlpha = 0.4;
    maskOverlayCtx.globalCompositeOperation = "source-atop";
    // Draw red where mask is white
    var maskData = maskCanvasCtx.canvas;
    maskOverlayCtx.drawImage(maskData, 0, 0);
    maskOverlayCtx.restore();

    // Simpler approach: draw red circles directly for visualization
    maskOverlayCtx.save();
    maskOverlayCtx.globalAlpha = 0.35;
    if (maskTool === "brush") {
      maskOverlayCtx.fillStyle = "red";
      maskOverlayCtx.beginPath();
      maskOverlayCtx.arc(x, y, radius, 0, Math.PI * 2);
      maskOverlayCtx.fill();
    }
    maskOverlayCtx.restore();
  }

  function getMaskBase64() {
    if (!maskCanvasCtx) return "";
    return maskCanvasCtx.canvas.toDataURL("image/png").split(",")[1];
  }

  function clearMask() {
    if (!maskCanvasCtx) return;
    var canvas = maskCanvasCtx.canvas;
    maskCanvasCtx.fillStyle = "black";
    maskCanvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    // Redraw overlay clean
    var visCanvas = $("#img-mask-canvas");
    if (visCanvas && visCanvas._sourceImg && maskOverlayCtx) {
      maskOverlayCtx.clearRect(0, 0, visCanvas.width, visCanvas.height);
      maskOverlayCtx.drawImage(visCanvas._sourceImg, 0, 0);
    }
  }

  function initImagePanel() {
    const toggleBtn = $("#toggle-image-btn");
    const panel = $("#image-panel");
    const closeBtn = $("#close-image-btn");
    const generateBtn = $("#img-generate-btn");
    const pullBtn = $("#img-pull-btn");
    const lightboxClose = $("#image-lightbox-close");
    const lightboxModal = $("#image-lightbox-modal");

    if (!toggleBtn || !panel) return;

    // Restore persisted image settings into form fields
    restoreImageSettings();

    // --- Image editing mode tabs ---
    document.querySelectorAll(".img-mode-tab").forEach(function (tab) {
      tab.addEventListener("click", function () {
        setImageMode(tab.dataset.mode);
      });
    });
    // Force correct initial visibility
    setImageMode("txt2img");

    // --- Source image upload ---
    var sourceDrop = $("#img-source-drop");
    var sourceFile = $("#img-source-file");
    if (sourceDrop) {
      sourceDrop.addEventListener("click", function () {
        if (sourceFile) sourceFile.click();
      });
      sourceDrop.addEventListener("dragenter", function (e) {
        e.preventDefault();
        e.stopPropagation();
        sourceDrop.classList.add("drag-over");
      });
      sourceDrop.addEventListener("dragover", function (e) {
        e.preventDefault();
        e.stopPropagation();
        sourceDrop.classList.add("drag-over");
      });
      sourceDrop.addEventListener("dragleave", function (e) {
        e.stopPropagation();
        sourceDrop.classList.remove("drag-over");
      });
      sourceDrop.addEventListener("drop", function (e) {
        e.preventDefault();
        e.stopPropagation();
        sourceDrop.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) {
          loadSourceImage(e.dataTransfer.files[0]);
        }
      });
    }
    if (sourceFile) {
      sourceFile.addEventListener("change", function () {
        if (sourceFile.files.length > 0) {
          loadSourceImage(sourceFile.files[0]);
        }
      });
    }

    // --- Strength slider ---
    var strengthSlider = $("#img-strength");
    var strengthVal = $("#img-strength-value");
    if (strengthSlider) {
      strengthSlider.addEventListener("input", function () {
        if (strengthVal) strengthVal.textContent = strengthSlider.value;
      });
    }

    // --- Mask tools ---
    document.querySelectorAll(".img-mask-tool").forEach(function (btn) {
      btn.addEventListener("click", function () {
        maskTool = btn.dataset.tool;
        document.querySelectorAll(".img-mask-tool").forEach(function (b) {
          b.classList.toggle("active", b.dataset.tool === maskTool);
        });
      });
    });

    var brushSizeEl = $("#img-mask-brush-size");
    if (brushSizeEl) {
      brushSizeEl.addEventListener("input", function () {
        maskBrushSize = parseInt(brushSizeEl.value) || 30;
      });
    }

    var clearMaskBtn = $("#img-mask-clear");
    if (clearMaskBtn) {
      clearMaskBtn.addEventListener("click", clearMask);
    }

    // --- Mask canvas pointer events ---
    var maskCanvas = $("#img-mask-canvas");
    if (maskCanvas) {
      maskCanvas.addEventListener("pointerdown", maskCanvasPointerDown);
      maskCanvas.addEventListener("pointermove", maskCanvasPointerMove);
      maskCanvas.addEventListener("pointerup", maskCanvasPointerUp);
      maskCanvas.addEventListener("pointerleave", maskCanvasPointerUp);
    }

    // --- Gallery collapse toggle ---
    var galleryToggle = $("#img-gallery-toggle");
    var galleryEl = $("#img-gallery");
    if (galleryToggle && galleryEl) {
      // Restore persisted state
      if (appSettings.galleryCollapsed) {
        galleryToggle.classList.add("collapsed");
        galleryEl.classList.add("collapsed");
      }
      galleryToggle.addEventListener("click", function () {
        var isCollapsed = galleryToggle.classList.toggle("collapsed");
        galleryEl.classList.toggle("collapsed", isCollapsed);
        appSettings.galleryCollapsed = isCollapsed;
        saveSettings();
      });
    }

    toggleBtn.addEventListener("click", function () {
      panel.classList.toggle("hidden");
      appSettings.imagePanelOpen = !panel.classList.contains("hidden");
      saveSettings();
      if (appSettings.imagePanelOpen) {
        fetchImageHardware();
        refreshImageModels();
        refreshImageGallery();
        refreshImageCatalog();
        fetchImageSamplers();
        renderResolutionPresets();
      }
    });

    // Update resolution presets when model changes
    var modelSelect = $("#img-model");
    if (modelSelect) {
      modelSelect.addEventListener("change", function () {
        var ptype = getSelectedPipelineType();
        var presets = IMG_RESOLUTION_PRESETS[ptype] || IMG_RESOLUTION_PRESETS.sd15;
        var widthEl = $("#img-width");
        var heightEl = $("#img-height");
        // Auto-set to first preset (native resolution) when switching models
        if (widthEl) widthEl.value = presets[0].w;
        if (heightEl) heightEl.value = presets[0].h;
        renderResolutionPresets();
      });
    }

    // Clear active preset highlight when manually editing width/height
    var widthEl = $("#img-width");
    var heightEl = $("#img-height");
    function clearPresetHighlight() {
      var container = $("#img-resolution-presets");
      if (container) container.querySelectorAll(".img-resolution-preset").forEach(function (b) { b.classList.remove("active"); });
    }
    if (widthEl) widthEl.addEventListener("input", clearPresetHighlight);
    if (heightEl) heightEl.addEventListener("input", clearPresetHighlight);

    // Persist image settings on any input change
    ["#img-width", "#img-height", "#img-steps", "#img-cfg", "#img-seed", "#img-negative"].forEach(function (sel) {
      var el = $(sel);
      if (el) el.addEventListener("input", saveImageSettings);
    });
    ["#img-sampler", "#img-model", "#img-preset", "#img-condense-model"].forEach(function (sel) {
      var el = $(sel);
      if (el) el.addEventListener("change", saveImageSettings);
    });


    // Unload model button
    var unloadBtn = $("#img-unload-btn");
    if (unloadBtn) {
      unloadBtn.addEventListener("click", async function () {
        try {
          var base = appSettings.backendUrl || "";
          var resp = await fetch(base + "/api/image/unload", { method: "POST" });
          var data = await resp.json();
          if (data.unloaded) {
            showToast("Model unloaded: " + data.model, "success");
          } else {
            showToast(data.reason || "No model loaded", "info");
          }
        } catch (err) {
          showToast("Unload failed: " + err.message, "error");
        }
      });
    }

    // Rename model button
    var renameBtn = $("#img-rename-btn");
    if (renameBtn) {
      renameBtn.addEventListener("click", handleImageModelRename);
    }

    // Expand/collapse prompt box
    var expandBtn = $("#img-expand-btn");
    if (expandBtn) {
      expandBtn.addEventListener("click", function () {
        var promptEl = $("#img-prompt");
        if (!promptEl) return;
        var expanded = promptEl.classList.toggle("expanded");
        expandBtn.classList.toggle("active", expanded);
        if (expanded) {
          promptEl.rows = 12;
        } else {
          promptEl.rows = 3;
        }
      });
    }

    // Enhance prompt button
    var enhanceBtn = $("#img-enhance-btn");
    if (enhanceBtn) {
      enhanceBtn.addEventListener("click", async function () {
        var promptEl = $("#img-prompt");
        if (!promptEl || !promptEl.value.trim()) {
          showToast("Enter a prompt first", "warning");
          return;
        }
        enhanceBtn.classList.add("enhancing");
        enhanceBtn.disabled = true;
        try {
          var base = appSettings.backendUrl || "";
          var condenseEl = $("#img-condense-model");
          var bodyObj = { prompt: promptEl.value };
          if (condenseEl && condenseEl.value) {
            bodyObj.model = condenseEl.value;
          }
          var resp = await fetch(base + "/api/image/enhance-prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(bodyObj),
          });
          if (!resp.ok) {
            var err = {};
            try { err = await resp.json(); } catch (_e) {}
            showToast(err.detail || "Enhancement failed", "error");
            return;
          }
          var data = await resp.json();
          if (data.prompt) {
            promptEl.value = data.prompt;
            showToast("Prompt enhanced", "success");
          }
        } catch (_e) {
          showToast("Enhancement failed", "error");
        } finally {
          enhanceBtn.classList.remove("enhancing");
          enhanceBtn.disabled = false;
        }
      });
    }

    // Generate negative prompt button
    var negGenBtn = $("#img-negative-gen-btn");
    if (negGenBtn) {
      negGenBtn.addEventListener("click", async function () {
        var promptEl = $("#img-prompt");
        if (!promptEl || !promptEl.value.trim()) {
          showToast("Enter a positive prompt first", "warning");
          return;
        }
        negGenBtn.classList.add("generating");
        negGenBtn.disabled = true;
        try {
          var base = appSettings.backendUrl || "";
          var condenseEl = $("#img-condense-model");
          var bodyObj = { prompt: promptEl.value };
          if (condenseEl && condenseEl.value) {
            bodyObj.model = condenseEl.value;
          }
          var resp = await fetch(base + "/api/image/generate-negative", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(bodyObj),
          });
          if (!resp.ok) {
            var err = {};
            try { err = await resp.json(); } catch (_e) {}
            showToast(err.detail || "Negative prompt generation failed", "error");
            return;
          }
          var data = await resp.json();
          if (data.negative_prompt) {
            var negEl = $("#img-negative");
            if (negEl) {
              negEl.value = data.negative_prompt;
              saveImageSettings();
            }
            showToast("Negative prompt generated", "success");
          }
        } catch (_e) {
          showToast("Negative prompt generation failed", "error");
        } finally {
          negGenBtn.classList.remove("generating");
          negGenBtn.disabled = false;
        }
      });
    }

    if (closeBtn) {
      closeBtn.addEventListener("click", function () {
        panel.classList.add("hidden");
        appSettings.imagePanelOpen = false;
        saveSettings();
      });
    }

    if (generateBtn) {
      generateBtn.addEventListener("click", handleImageGenerate);
    }

    var cancelBtn = $("#img-cancel-btn");
    if (cancelBtn) {
      cancelBtn.addEventListener("click", async function () {
        // Abort the fetch and tell the server to cancel the running job
        if (imageAbortController) imageAbortController.abort();
        try {
          var base = appSettings.backendUrl || "";
          await fetch(base + "/api/image/cancel", { method: "POST" });
        } catch (_e) { /* best effort */ }
      });
    }

    if (pullBtn) {
      pullBtn.addEventListener("click", handleImagePull);
    }

    if (lightboxClose) {
      lightboxClose.addEventListener("click", closeLightbox);
    }

    if (lightboxModal) {
      lightboxModal.addEventListener("click", function (e) {
        if (e.target === lightboxModal) closeLightbox();
      });
    }

    // --- Lightbox edit buttons ---
    var editImg2imgBtn = $("#lightbox-edit-img2img");
    var editInpaintBtn = $("#lightbox-edit-inpaint");

    function sendLightboxToEdit(mode) {
      var imgEl = $("#lightbox-img");
      if (!imgEl || !imgEl.src) return;
      closeLightbox();
      setImageMode(mode);
      if (panel) panel.classList.remove("hidden");
      loadSourceImageFromUrl(imgEl.src);
    }

    if (editImg2imgBtn) {
      editImg2imgBtn.addEventListener("click", function () {
        sendLightboxToEdit("img2img");
      });
    }
    if (editInpaintBtn) {
      editInpaintBtn.addEventListener("click", function () {
        sendLightboxToEdit("inpaint");
      });
    }

    // --- Lightbox action buttons ---
    var lbCopyPrompt = $("#lightbox-copy-prompt");
    var lbCopySeed = $("#lightbox-copy-seed");
    var lbDownload = $("#lightbox-download");
    var lbUsePrompt = $("#lightbox-use-prompt");
    var lbDelete = $("#lightbox-delete");

    if (lbCopyPrompt) {
      lbCopyPrompt.addEventListener("click", function () {
        if (!currentLightboxEntry || !currentLightboxEntry.prompt) return;
        navigator.clipboard.writeText(currentLightboxEntry.prompt).then(function () {
          showToast("Prompt copied", "success");
        });
      });
    }

    if (lbCopySeed) {
      lbCopySeed.addEventListener("click", function () {
        if (!currentLightboxEntry || currentLightboxEntry.seed == null) return;
        navigator.clipboard.writeText(String(currentLightboxEntry.seed)).then(function () {
          showToast("Seed copied: " + currentLightboxEntry.seed, "success");
        });
      });
    }

    if (lbDownload) {
      lbDownload.addEventListener("click", function () {
        var imgEl = $("#lightbox-img");
        if (!imgEl || !imgEl.src) return;
        var a = document.createElement("a");
        a.href = imgEl.src;
        a.download = (currentLightboxEntry && currentLightboxEntry.image_id ? currentLightboxEntry.image_id : "image") + ".png";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      });
    }

    if (lbUsePrompt) {
      lbUsePrompt.addEventListener("click", function () {
        if (!currentLightboxEntry) return;
        var e = currentLightboxEntry;
        var promptEl = $("#img-prompt");
        if (promptEl && e.prompt) promptEl.value = e.prompt;
        var negEl = $("#img-negative");
        if (negEl && e.negative_prompt) negEl.value = e.negative_prompt;
        var seedEl = $("#img-seed");
        if (seedEl && e.seed != null) seedEl.value = e.seed;
        if (e.width) { var w = $("#img-width"); if (w) w.value = e.width; }
        if (e.height) { var h = $("#img-height"); if (h) h.value = e.height; }
        if (e.steps) { var s = $("#img-steps"); if (s) s.value = e.steps; }
        if (e.cfg_scale) { var c = $("#img-cfg"); if (c) c.value = e.cfg_scale; }
        saveImageSettings();
        closeLightbox();
        if (panel) panel.classList.remove("hidden");
        showToast("Settings loaded from image", "success");
      });
    }

    if (lbDelete) {
      lbDelete.addEventListener("click", async function () {
        if (!currentLightboxEntry || !currentLightboxEntry.image_id) return;
        if (!confirm("Delete this image?")) return;
        try {
          var base = appSettings.backendUrl || "";
          var resp = await fetch(base + "/api/image/" + currentLightboxEntry.image_id, { method: "DELETE" });
          if (resp.ok) {
            showToast("Image deleted", "success");
            closeLightbox();
            refreshImageGallery();
          } else {
            showToast("Failed to delete image", "error");
          }
        } catch (err) {
          showToast("Delete failed: " + err.message, "error");
        }
      });
    }
  }

  async function handleImageGenerate() {
    if (imageGenerating) return;

    var promptEl = $("#img-prompt");
    var prompt = promptEl ? promptEl.value : "";
    if (!prompt.trim()) return;

    var negEl = $("#img-negative");
    var negative = negEl ? negEl.value : "";
    var presetEl = $("#img-preset");
    var preset = presetEl ? presetEl.value : "";
    var widthEl = $("#img-width");
    var width = widthEl ? parseInt(widthEl.value) || 512 : 512;
    var heightEl = $("#img-height");
    var height = heightEl ? parseInt(heightEl.value) || 512 : 512;
    var stepsEl = $("#img-steps");
    var steps = stepsEl ? parseInt(stepsEl.value) || 20 : 20;
    var cfgEl = $("#img-cfg");
    var cfg = cfgEl ? parseFloat(cfgEl.value) || 7.0 : 7.0;
    var seedEl = $("#img-seed");
    var seed = seedEl ? parseInt(seedEl.value) || -1 : -1;
    var samplerEl = $("#img-sampler");
    var sampler = samplerEl ? samplerEl.value : "";
    var modelEl = $("#img-model");
    var model = modelEl ? modelEl.value : "";

    var generateBtn = $("#img-generate-btn");
    var progress = $("#img-progress");

    // Pre-generate VRAM courtesy check
    if (imageHardwareInfo && model) {
      var selectedModel = imageModelsData.find(function (m) { return m.name === model; });
      if (selectedModel) {
        var vramNeeded = { sd15: 2000, sdxl: 5500, flux: 10000 };
        var ptype = selectedModel.pipeline_type;
        var needed = vramNeeded[ptype] || 0;
        var freeMb = imageHardwareInfo.vram_free_mb || 0;
        var isCpu = imageHardwareInfo.device === "cpu";
        var warn = false;
        if (isCpu && ptype !== "sd15") {
          warn = true;
        } else if (!isCpu && needed > 0 && freeMb < needed) {
          warn = true;
        }
        if (warn) {
          if (!confirm("This model may exceed your GPU memory. Continue anyway?")) return;
        }
      }
    }

    imageGenerating = true;
    imageAbortController = new AbortController();
    var cancelBtn = $("#img-cancel-btn");
    if (generateBtn) generateBtn.classList.add("hidden");
    if (cancelBtn) cancelBtn.classList.remove("hidden");
    if (progress) progress.classList.remove("hidden");

    try {
      var base = appSettings.backendUrl || "";
      var endpoint, bodyObj;

      if (currentImageMode === "img2img") {
        if (!sourceImageBase64) {
          showToast("Please upload a source image first", "error");
          return;
        }
        var strengthEl2 = $("#img-strength");
        var strength = strengthEl2 ? parseFloat(strengthEl2.value) || 0.75 : 0.75;
        endpoint = "/api/image/img2img";
        bodyObj = {
          prompt: prompt,
          negative_prompt: negative,
          model: model,
          source_image: sourceImageBase64,
          strength: strength,
          width: width,
          height: height,
          steps: steps,
          cfg_scale: cfg,
          seed: seed,
          sampler: sampler || undefined,
          preset: preset,
        };
      } else if (currentImageMode === "inpaint") {
        if (!sourceImageBase64) {
          showToast("Please upload a source image first", "error");
          return;
        }
        var maskB64 = getMaskBase64();
        if (!maskB64) {
          showToast("Please paint a mask on the image", "error");
          return;
        }
        var strengthEl3 = $("#img-strength");
        var strengthInp = strengthEl3 ? parseFloat(strengthEl3.value) || 1.0 : 1.0;
        endpoint = "/api/image/inpaint";
        bodyObj = {
          prompt: prompt,
          negative_prompt: negative,
          model: model,
          source_image: sourceImageBase64,
          mask_image: maskB64,
          strength: strengthInp,
          width: width,
          height: height,
          steps: steps,
          cfg_scale: cfg,
          seed: seed,
          sampler: sampler || undefined,
          preset: preset,
        };
      } else {
        endpoint = "/api/image/generate";
        bodyObj = {
          prompt: prompt,
          negative_prompt: negative,
          preset: preset,
          width: width,
          height: height,
          steps: steps,
          cfg_scale: cfg,
          seed: seed,
          sampler: sampler || undefined,
          model: model,
        };
      }

      // Attach condense model preference
      var condenseEl = $("#img-condense-model");
      if (condenseEl && condenseEl.value) {
        bodyObj.condense_model = condenseEl.value;
      }

      var resp = await fetch(base + endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(bodyObj),
        signal: imageAbortController.signal,
      });

      if (!resp.ok) {
        var err = {};
        try { err = await resp.json(); } catch (_e) { err = { error: "Generation failed" }; }
        var msg = err.detail || err.error || "Image generation failed";
        showToast(msg, "error");
        return;
      }

      var data = await resp.json();
      refreshImageGallery();

      if (data.url) {
        var base2 = appSettings.backendUrl || "";
        var fullUrl2 = data.url.startsWith("http") ? data.url : base2 + data.url;
        openLightbox({
          image_id: data.image_id || "",
          prompt: data.prompt || prompt,
          negative_prompt: data.negative_prompt || negative,
          seed: data.seed || -1,
          width: data.width || width,
          height: data.height || height,
          steps: data.steps || steps,
          cfg_scale: data.cfg_scale || cfg,
          model: data.model || model,
        }, fullUrl2);
      }
    } catch (err) {
      if (err.name === "AbortError") {
        showToast("Image generation cancelled", "warning");
      } else {
        showToast("Image generation failed: " + err.message, "error");
      }
    } finally {
      imageGenerating = false;
      imageAbortController = null;
      var cancelBtn2 = $("#img-cancel-btn");
      if (generateBtn) generateBtn.classList.remove("hidden");
      if (cancelBtn2) cancelBtn2.classList.add("hidden");
      if (progress) progress.classList.add("hidden");
    }
  }

  async function refreshImageGallery() {
    var gallery = $("#img-gallery");
    if (!gallery) return;

    try {
      var base = appSettings.backendUrl || "";
      var resp = await fetch(base + "/api/image/history?limit=20");
      if (!resp.ok) return;
      var data = await resp.json();
      var entries = Array.isArray(data) ? data : (data.entries || []);

      if (!entries || entries.length === 0) {
        gallery.innerHTML =
          '<div class="empty-state">' +
          '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="32" height="32"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>' +
          "<p>Generated images will appear here.</p></div>";
        return;
      }

      var countEl = $("#img-gallery-count");
      if (countEl) countEl.textContent = "(" + entries.length + ")";

      gallery.innerHTML = "";
      for (var i = 0; i < entries.length; i++) {
        var entry = entries[i];
        var url = (appSettings.backendUrl || "") + (entry.url || "/api/image/" + entry.image_id);
        var thumb = document.createElement("div");
        thumb.className = "img-thumb";
        var img = document.createElement("img");
        img.src = url;
        img.alt = entry.prompt || "";
        img.loading = "lazy";
        thumb.appendChild(img);
        (function (entryRef, urlRef) {
          thumb.addEventListener("click", function () { openLightbox(entryRef, urlRef); });
        })(entry, url);
        gallery.appendChild(thumb);
      }
    } catch (_e) {
      // Silently fail if image API is not available
    }
  }

  var currentLightboxEntry = null;

  function openLightbox(entryOrUrl, urlOverride) {
    var modal = $("#image-lightbox-modal");
    var imgEl = $("#lightbox-img");
    var meta = $("#lightbox-meta");

    if (!modal || !imgEl) return;

    // Support both old-style (url, prompt, seed, w, h) and new-style (entry, url)
    var entry, url;
    if (typeof entryOrUrl === "string") {
      // Legacy call: openLightbox(url, prompt, seed, w, h)
      url = entryOrUrl;
      entry = {
        prompt: urlOverride || "",
        seed: arguments[2] || -1,
        width: arguments[3] || 0,
        height: arguments[4] || 0,
        image_id: "",
      };
    } else {
      entry = entryOrUrl;
      url = urlOverride || entry.url || "";
    }

    currentLightboxEntry = entry;

    var fullUrl = url.startsWith("http") ? url : (appSettings.backendUrl || "") + url;
    imgEl.src = fullUrl;
    if (meta) {
      var parts = [];
      if (entry.prompt) parts.push(entry.prompt.substring(0, 200));
      if (entry.seed && entry.seed !== -1) parts.push("Seed: " + entry.seed);
      if (entry.width && entry.height) parts.push(entry.width + "x" + entry.height);
      if (entry.model) parts.push(entry.model);
      if (entry.steps) parts.push(entry.steps + " steps");
      meta.textContent = parts.join(" \u2022 ");
    }
    modal.classList.add("visible");
  }

  function closeLightbox() {
    var modal = $("#image-lightbox-modal");
    if (modal) modal.classList.remove("visible");
    currentLightboxEntry = null;
  }

  async function refreshImageModels() {
    var select = $("#img-model");
    if (!select) return;

    try {
      var base = appSettings.backendUrl || "";
      var resp = await fetch(base + "/api/image/models");
      if (!resp.ok) return;
      var models = await resp.json();

      imageModelsData = models;
      select.innerHTML = '<option value="">Default</option>';
      for (var i = 0; i < models.length; i++) {
        var m = models[i];
        var opt = document.createElement("option");
        opt.value = m.name;
        opt.textContent = m.name + " (" + m.pipeline_type + ")";
        if (m.is_loaded) opt.textContent += " *";
        select.appendChild(opt);
      }
      // Restore persisted model selection
      if (appSettings.imgModel) {
        select.value = appSettings.imgModel;
      }
      renderResolutionPresets();
    } catch (_e) {
      // Image models endpoint not available
    }
  }

  async function handleImageModelRename() {
    var modelEl = $("#img-model");
    var oldName = modelEl ? modelEl.value : "";
    if (!oldName) {
      showToast("Select a model to rename", "warning");
      return;
    }

    var newName = prompt("Enter new name for \"" + oldName + "\":", oldName);
    if (!newName || newName.trim() === "" || newName.trim() === oldName) return;
    newName = newName.trim();

    try {
      var base = appSettings.backendUrl || "";
      var resp = await fetch(base + "/api/image/models/rename", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ old_name: oldName, new_name: newName }),
      });
      if (!resp.ok) {
        var err = {};
        try { err = await resp.json(); } catch (_) {}
        showToast(err.detail || "Rename failed", "error");
        return;
      }
      showToast("Renamed to \"" + newName + "\"", "success");
      refreshImageModels();
    } catch (err) {
      showToast("Rename failed: " + err.message, "error");
    }
  }

  // --- Shared polling helper for background model downloads ---

  function pollPullTask(taskId, onProgress, onComplete, onError) {
    var base = appSettings.backendUrl || "";
    var consecutiveErrors = 0;
    var maxRetries = 10;  // Allow up to 10 consecutive poll failures before giving up
    var interval = setInterval(async function () {
      try {
        var resp = await fetch(base + "/api/image/models/pull/" + taskId);
        if (!resp.ok) {
          if (resp.status === 404) {
            // Task not found — may have been cleaned up or never existed
            clearInterval(interval);
            onError("Download task not found (may have completed)");
            return;
          }
          consecutiveErrors++;
          if (consecutiveErrors >= maxRetries) {
            clearInterval(interval);
            onError("Lost connection to download task after " + maxRetries + " retries");
          }
          return;
        }
        var data = await resp.json();
        consecutiveErrors = 0;  // Reset on success

        if (data.status === "running") {
          onProgress(data);
        } else if (data.status === "complete") {
          clearInterval(interval);
          onComplete(data);
        } else if (data.status === "exists") {
          clearInterval(interval);
          onComplete(data);
        } else if (data.status === "error") {
          clearInterval(interval);
          onError(data.error || "Unknown error");
        }
      } catch (err) {
        consecutiveErrors++;
        if (consecutiveErrors >= maxRetries) {
          clearInterval(interval);
          onError("Polling failed after " + maxRetries + " retries: " + err.message);
        }
        // Otherwise silently retry on next interval
      }
    }, 2000);
    return interval;
  }

  async function handleImagePull() {
    var input = $("#img-pull-input");
    var progressArea = $("#img-pull-progress");
    var fill = $("#img-pull-fill");
    var statusEl = $("#img-pull-status");

    if (!input || !input.value.trim()) return;

    var source = input.value.trim();
    if (progressArea) progressArea.classList.remove("hidden");
    if (statusEl) statusEl.textContent = "Starting download...";
    if (fill) fill.style.width = "0%";

    try {
      var base = appSettings.backendUrl || "";
      var resp = await fetch(base + "/api/image/models/pull", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: source }),
      });

      if (!resp.ok) {
        var errBody = "";
        try { var ej = await resp.json(); errBody = ej.detail || ej.error || ""; } catch (_) {}
        if (statusEl) statusEl.textContent = "Error: " + (errBody || "HTTP " + resp.status);
        setTimeout(function () { if (progressArea) progressArea.classList.add("hidden"); }, 4000);
        return;
      }

      var task = await resp.json();
      if (statusEl) statusEl.textContent = "Downloading...";

      pollPullTask(task.task_id,
        function onProgress(data) {
          if (data.percent !== undefined) {
            var pct = Math.round(data.percent);
            if (fill) fill.style.width = Math.max(pct, 2) + "%";
            if (pct === 0 && !data.downloaded) {
              if (statusEl) statusEl.textContent = "Preparing download...";
            } else {
              var pctText = pct + "%";
              if (data.downloaded && data.total) {
                pctText += " — " + formatBytes(data.downloaded) + " / " + formatBytes(data.total);
              }
              if (statusEl) statusEl.textContent = pctText;
            }
          }
        },
        function onComplete(data) {
          if (fill) fill.style.width = "100%";
          if (statusEl) statusEl.textContent = data.status === "exists" ? "Model already exists." : "Complete!";
          refreshImageModels();
          refreshImageCatalog();
          setTimeout(function () { if (progressArea) progressArea.classList.add("hidden"); }, 3000);
        },
        function onError(errMsg) {
          if (statusEl) statusEl.textContent = "Error: " + errMsg;
          setTimeout(function () { if (progressArea) progressArea.classList.add("hidden"); }, 6000);
        }
      );
    } catch (err) {
      if (statusEl) statusEl.textContent = "Download failed: " + err.message;
      setTimeout(function () { if (progressArea) progressArea.classList.add("hidden"); }, 4000);
    }
  }

  async function refreshImageCatalog() {
    var container = $("#img-catalog");
    if (!container) return;

    try {
      var base = appSettings.backendUrl || "";
      var resp = await fetch(base + "/api/image/models/catalog");
      if (!resp.ok) {
        container.innerHTML = '<div class="empty-state"><p>Catalog unavailable</p></div>';
        return;
      }
      var catalog = await resp.json();

      if (!catalog || catalog.length === 0) {
        container.innerHTML = '<div class="empty-state"><p>No models in catalog</p></div>';
        return;
      }

      container.innerHTML = "";
      for (var i = 0; i < catalog.length; i++) {
        var m = catalog[i];
        container.appendChild(buildCatalogCard(m));
      }
      // Reconnect any in-progress downloads to their catalog buttons
      reconnectCatalogDownloads();
    } catch (_e) {
      container.innerHTML = '<div class="empty-state"><p>Could not load catalog</p></div>';
    }
  }

  function buildCatalogCard(model) {
    var card = document.createElement("div");
    card.className = "catalog-card" + (model.compatible ? "" : " incompatible");
    var isGguf = !!(model.allow_patterns && model.allow_patterns.length);

    // VRAM display
    var vramText = model.min_vram_mb > 0
      ? (model.min_vram_mb >= 1000 ? (model.min_vram_mb / 1000) + "GB VRAM" : model.min_vram_mb + "MB VRAM")
      : "No GPU required";

    // Badges
    var badges = '<span class="catalog-badge pipeline-' + model.pipeline_type + '">' + model.pipeline_type.toUpperCase() + '</span>';
    if (isGguf) badges += '<span class="catalog-badge gguf-tag">GGUF</span>';
    if (model.cpu_friendly) {
      badges += '<span class="catalog-badge cpu-ok">CPU OK</span>';
    }
    if (model.installed) {
      badges += '<span class="catalog-badge installed">Installed</span>';
    }

    // Download button
    var btnClass = "catalog-download-btn";
    var btnText = "Download";
    var btnDisabled = "";
    if (model.installed) {
      btnClass += " installed-btn";
      btnText = "Installed";
      btnDisabled = " disabled";
    } else if (!model.compatible) {
      btnText = "Download";
    }

    // GGUF quant selector row
    var quantRow = isGguf && !model.installed
      ? '<div class="catalog-quant-row">' +
          '<select class="catalog-quant-select" data-repo="' + escapeHtml(model.repo_id) + '">' +
            '<option value="" disabled selected>Select quantization…</option>' +
          '</select>' +
          '<span class="catalog-quant-size"></span>' +
        '</div>'
      : '';

    card.innerHTML =
      '<div class="catalog-card-header">' +
        '<span class="catalog-card-name">' + escapeHtml(model.name) + '</span>' +
        '<div class="catalog-card-badges">' + badges + '</div>' +
      '</div>' +
      '<div class="catalog-card-desc">' + escapeHtml(model.description) + '</div>' +
      quantRow +
      '<div class="catalog-card-meta">' +
        '<div class="catalog-card-specs">' +
          '<span class="catalog-spec">' +
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 12h4m4 0h4"/></svg>' +
            vramText +
          '</span>' +
          '<span class="catalog-spec catalog-size-label">' +
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>' +
            model.size_gb + ' GB' +
          '</span>' +
          (model.speed_note ? '<span class="catalog-spec">' +
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>' +
            escapeHtml(model.speed_note) +
          '</span>' : '') +
        '</div>' +
        '<button class="' + btnClass + '"' + btnDisabled + ' data-repo="' + escapeHtml(model.repo_id) + '">' + btnText + '</button>' +
      '</div>';

    if (!model.installed) {
      var btn = card.querySelector(".catalog-download-btn");

      if (isGguf) {
        var select = card.querySelector(".catalog-quant-select");
        var sizeLabel = card.querySelector(".catalog-quant-size");
        var cardSizeLabel = card.querySelector(".catalog-size-label");
        var variantsLoaded = false;
        var variants = [];

        function loadVariants() {
          if (variantsLoaded) return;
          variantsLoaded = true;
          select.innerHTML = '<option value="" disabled selected>Loading…</option>';
          var base = appSettings.backendUrl || "";
          fetch(base + "/api/image/models/variants?repo_id=" + encodeURIComponent(model.repo_id))
            .then(function (r) { return r.json(); })
            .then(function (data) {
              variants = data.variants || [];
              select.innerHTML = "";
              if (!variants.length) {
                select.innerHTML = '<option value="" disabled selected>No variants found</option>';
                return;
              }
              var defaultQuant = model.allow_patterns[0].replace(/\*/g, "");
              for (var i = 0; i < variants.length; i++) {
                var v = variants[i];
                var opt = document.createElement("option");
                opt.value = v.pattern;
                opt.textContent = v.quant + (v.size_gb ? " (" + v.size_gb + " GB)" : "");
                opt.dataset.sizeGb = v.size_gb;
                if (v.quant === defaultQuant) opt.selected = true;
                select.appendChild(opt);
              }
              if (select.value) {
                var sel = variants.find(function (v) { return v.pattern === select.value; });
                if (sel) {
                  sizeLabel.textContent = sel.size_gb + " GB";
                  cardSizeLabel.textContent = sel.size_gb + " GB";
                }
              }
            })
            .catch(function () {
              select.innerHTML = '<option value="" disabled selected>Failed to load</option>';
            });
        }

        select.addEventListener("focus", loadVariants);
        select.addEventListener("mousedown", loadVariants);
        select.addEventListener("change", function () {
          var sel = variants.find(function (v) { return v.pattern === select.value; });
          if (sel) {
            sizeLabel.textContent = sel.size_gb + " GB";
            cardSizeLabel.textContent = sel.size_gb + " GB";
          }
        });

        btn.addEventListener("click", function () {
          var pattern = select.value;
          if (!pattern) {
            loadVariants();
            select.focus();
            return;
          }
          handleCatalogDownload(model.repo_id, btn, [pattern]);
        });
      } else {
        btn.addEventListener("click", function () {
          handleCatalogDownload(model.repo_id, btn, model.allow_patterns || null);
        });
      }
    }

    return card;
  }

  function escapeHtml(str) {
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  // Track active catalog downloads so we can reconnect on page load
  var _activeCatalogDownloads = {};  // repoId → { taskId, btn, interval }

  async function handleCatalogDownload(repoId, btn, allowPatterns) {
    if (btn.disabled) return;

    btn.disabled = true;
    btn.classList.add("downloading");
    var originalText = btn.textContent;
    btn.textContent = "Starting...";

    function resetBtn(text, delay) {
      btn.textContent = text;
      setTimeout(function () {
        btn.textContent = originalText;
        btn.disabled = false;
        btn.classList.remove("downloading");
      }, delay);
      delete _activeCatalogDownloads[repoId];
    }

    try {
      var base = appSettings.backendUrl || "";
      var body = { source: repoId };
      if (allowPatterns) body.allow_patterns = allowPatterns;
      var resp = await fetch(base + "/api/image/models/pull", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!resp.ok) {
        var errBody = "";
        try { var ej = await resp.json(); errBody = ej.detail || ej.error || ""; } catch (_) {}
        resetBtn("Error: " + (errBody || resp.status), 6000);
        return;
      }

      var task = await resp.json();
      btn.textContent = "Preparing...";

      _activeCatalogDownloads[repoId] = { taskId: task.task_id, btn: btn };

      var interval = pollPullTask(task.task_id,
        function onProgress(data) {
          if (data.percent !== undefined) {
            var pct = Math.round(data.percent);
            if (pct === 0 && !data.downloaded) {
              btn.textContent = "Preparing...";
            } else {
              var detail = pct + "%";
              if (data.downloaded && data.total) {
                detail += " " + formatBytes(data.downloaded) + "/" + formatBytes(data.total);
              }
              btn.textContent = detail;
            }
          }
        },
        function onComplete(data) {
          btn.textContent = "Installed";
          btn.classList.remove("downloading");
          btn.classList.add("installed-btn");
          delete _activeCatalogDownloads[repoId];
          refreshImageModels();
          refreshImageCatalog();
        },
        function onError(errMsg) {
          resetBtn("Error: " + errMsg.substring(0, 40), 6000);
        }
      );
      _activeCatalogDownloads[repoId].interval = interval;
    } catch (err) {
      resetBtn("Failed: " + err.message.substring(0, 40), 6000);
    }
  }

  async function reconnectCatalogDownloads() {
    // Check for any in-progress downloads on the server and reconnect UI
    try {
      var base = appSettings.backendUrl || "";
      var resp = await fetch(base + "/api/image/models/pull");
      if (!resp.ok) return;
      var tasks = await resp.json();
      if (!tasks || tasks.length === 0) return;

      for (var i = 0; i < tasks.length; i++) {
        var t = tasks[i];
        if (t.status !== "running") continue;
        var source = t.source || "";
        if (!source) continue;

        // Already tracking this download
        if (_activeCatalogDownloads[source]) continue;

        // Find the catalog card button for this source
        var catalogBtn = document.querySelector('.catalog-download-btn[data-repo="' + source + '"]');
        if (!catalogBtn) continue;

        // Reconnect the button to show download progress
        catalogBtn.disabled = true;
        catalogBtn.classList.add("downloading");
        catalogBtn.textContent = t.percent ? Math.round(t.percent) + "%" : "Downloading...";

        _activeCatalogDownloads[source] = { taskId: t.task_id, btn: catalogBtn };

        (function (repo, btn, taskId) {
          var interval = pollPullTask(taskId,
            function onProgress(data) {
              if (data.percent !== undefined) {
                var pct = Math.round(data.percent);
                if (pct === 0 && !data.downloaded) {
                  btn.textContent = "Preparing...";
                } else {
                  var detail = pct + "%";
                  if (data.downloaded && data.total) {
                    detail += " " + formatBytes(data.downloaded) + "/" + formatBytes(data.total);
                  }
                  btn.textContent = detail;
                }
              }
            },
            function onComplete() {
              btn.textContent = "Installed";
              btn.classList.remove("downloading");
              btn.classList.add("installed-btn");
              delete _activeCatalogDownloads[repo];
              refreshImageModels();
              refreshImageCatalog();
            },
            function onError() {
              btn.textContent = "Download";
              btn.disabled = false;
              btn.classList.remove("downloading");
              delete _activeCatalogDownloads[repo];
            }
          );
          _activeCatalogDownloads[repo].interval = interval;
        })(source, catalogBtn, t.task_id);
      }
    } catch (_) {
      // Not critical — downloads will still work, just won't show progress
    }
  }

  // ---- Image Library ----

  var imgLibState = {
    view: "grid",
    bulkMode: false,
    selectedIds: new Set(),
    offset: 0,
    total: 0,
    currentEntry: null,
    searchTimer: null,
    loading: false,
    entries: [],
  };

  function initImageLibrary() {
    var openBtn = $("#open-image-library-btn");
    var closeBtn = $("#img-library-close");
    var modal = $("#img-library-modal");
    var searchInput = $("#img-lib-search");
    var modelFilter = $("#img-lib-model-filter");
    var presetFilter = $("#img-lib-preset-filter");
    var sortSelect = $("#img-lib-sort");
    var gridBtn = $("#img-lib-view-grid");
    var listBtn = $("#img-lib-view-list");
    var loadMoreBtn = $("#img-lib-load-more-btn");
    var selectBtn = $("#img-lib-select-btn");
    var bulkDeleteBtn = $("#img-lib-bulk-delete-btn");
    var detailClose = $("#img-lib-detail-close");
    var usePromptBtn = $("#img-lib-use-prompt");
    var downloadBtn = $("#img-lib-download");
    var copySeedBtn = $("#img-lib-copy-seed");
    var deleteBtn = $("#img-lib-delete");

    if (!openBtn || !modal) return;

    openBtn.addEventListener("click", openImageLibrary);
    closeBtn.addEventListener("click", closeImageLibrary);
    modal.addEventListener("click", function (e) {
      if (e.target === modal) closeImageLibrary();
    });

    searchInput.addEventListener("input", function () {
      clearTimeout(imgLibState.searchTimer);
      imgLibState.searchTimer = setTimeout(imgLibResetAndLoad, 350);
    });

    modelFilter.addEventListener("change", imgLibResetAndLoad);
    presetFilter.addEventListener("change", imgLibResetAndLoad);
    sortSelect.addEventListener("change", imgLibResetAndLoad);

    gridBtn.addEventListener("click", function () { setLibraryView("grid"); });
    listBtn.addEventListener("click", function () { setLibraryView("list"); });

    loadMoreBtn.addEventListener("click", imgLibLoadMore);
    selectBtn.addEventListener("click", toggleBulkMode);
    bulkDeleteBtn.addEventListener("click", handleBulkDelete);
    detailClose.addEventListener("click", closeLibraryDetail);
    usePromptBtn.addEventListener("click", handleLibraryReusePrompt);
    downloadBtn.addEventListener("click", handleLibraryDownload);
    copySeedBtn.addEventListener("click", handleLibraryCopySeed);
    deleteBtn.addEventListener("click", function () { handleLibraryDelete(); });

    // Grid click delegation
    var gridEl = $("#img-lib-grid");
    gridEl.addEventListener("click", function (e) {
      var actionBtn = e.target.closest(".img-lib-card-action");
      if (actionBtn) {
        e.stopPropagation();
        var action = actionBtn.dataset.action;
        var id = actionBtn.closest("[data-image-id]").dataset.imageId;
        var entry = imgLibState.entries.find(function (en) { return en.image_id === id; });
        if (!entry) return;
        if (action === "download") {
          imgLibState.currentEntry = entry;
          handleLibraryDownload();
        } else if (action === "delete") {
          handleLibraryDelete(id);
        }
        return;
      }

      var checkEl = e.target.closest(".img-lib-card-check, .img-lib-row-check");
      if (checkEl && imgLibState.bulkMode) {
        e.stopPropagation();
        var card = checkEl.closest("[data-image-id]");
        var imgId = card.dataset.imageId;
        if (imgLibState.selectedIds.has(imgId)) {
          imgLibState.selectedIds.delete(imgId);
          card.classList.remove("selected");
          checkEl.innerHTML = "";
        } else {
          imgLibState.selectedIds.add(imgId);
          card.classList.add("selected");
          checkEl.innerHTML = "&#10003;";
        }
        updateBulkDeleteLabel();
        return;
      }

      var cardEl = e.target.closest("[data-image-id]");
      if (cardEl) {
        if (imgLibState.bulkMode) {
          var chk = cardEl.querySelector(".img-lib-card-check, .img-lib-row-check");
          if (chk) chk.click();
          return;
        }
        var entryId = cardEl.dataset.imageId;
        var entry2 = imgLibState.entries.find(function (en) { return en.image_id === entryId; });
        if (entry2) imgLibSelectEntry(entry2);
      }
    });

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && modal.classList.contains("visible")) {
        var detail = $("#img-lib-detail");
        if (detail && !detail.classList.contains("hidden")) {
          closeLibraryDetail();
        } else {
          closeImageLibrary();
        }
      }
    });

    // Tab switching
    var tabsEl = $("#img-lib-tabs");
    if (tabsEl) {
      tabsEl.addEventListener("click", function (e) {
        var tab = e.target.closest(".img-lib-tab");
        if (!tab) return;
        var tabName = tab.dataset.tab;
        tabsEl.querySelectorAll(".img-lib-tab").forEach(function (t) {
          t.classList.toggle("active", t.dataset.tab === tabName);
        });
        document.querySelectorAll(".img-lib-tab-content").forEach(function (c) {
          c.classList.toggle("active", c.id === "img-lib-tab-" + tabName);
        });
        if (tabName === "models") imgLibLoadModels();
        if (tabName === "presets") imgLibLoadPresets();
      });
    }

    // Model pull
    var pullBtn = $("#img-lib-model-pull");
    if (pullBtn) {
      pullBtn.addEventListener("click", imgLibPullModel);
    }

    // Model drag-and-drop upload
    initModelDropzone();

    // Preset create
    var presetCreateBtn = $("#img-lib-preset-create");
    if (presetCreateBtn) {
      presetCreateBtn.addEventListener("click", imgLibCreatePreset);
    }
  }

  function openImageLibrary() {
    var modal = $("#img-library-modal");
    modal.classList.add("visible");
    imgLibState.bulkMode = false;
    imgLibState.selectedIds.clear();
    imgLibState.currentEntry = null;
    imgLibState.entries = [];
    var grid = $("#img-lib-grid");
    grid.classList.remove("bulk-mode");
    $("#img-lib-bulk-delete-btn").classList.add("hidden");
    $("#img-lib-select-btn").textContent = "Select";
    closeLibraryDetail();
    populateLibraryFilters();
    imgLibResetAndLoad();
  }

  function closeImageLibrary() {
    var modal = $("#img-library-modal");
    modal.classList.remove("visible");
    clearTimeout(imgLibState.searchTimer);
  }

  // ---- Image Models Management ----

  async function imgLibLoadModels() {
    var list = $("#img-lib-models-list");
    if (!list) return;
    var base = appSettings.backendUrl || "";
    try {
      var resp = await fetch(base + "/api/image/models");
      if (!resp.ok) { list.innerHTML = '<div class="img-lib-empty-state">Could not load models</div>'; return; }
      var models = await resp.json();
      if (models.length === 0) {
        list.innerHTML = '<div class="img-lib-empty-state">No models installed. Download one below.</div>';
        return;
      }
      list.innerHTML = models.map(function (m) {
        var sizeStr = m.size_bytes ? (m.size_bytes / 1024 / 1024 / 1024).toFixed(1) + " GB" : "";
        var loadedBadge = m.is_loaded ? '<span class="img-lib-model-badge loaded">Loaded</span>' : "";
        var typeBadge = '<span class="img-lib-model-badge type">' + escapeHtml(m.pipeline_type) + "</span>";
        return '<div class="img-lib-model-card" data-model-name="' + escapeHtml(m.name) + '">' +
          '<div class="img-lib-model-info">' +
            '<div class="img-lib-model-name">' + escapeHtml(m.name) + "</div>" +
            '<div class="img-lib-model-meta">' + escapeHtml(m.source || "") + (sizeStr ? " &middot; " + sizeStr : "") + "</div>" +
          "</div>" +
          typeBadge + loadedBadge +
          '<div class="img-lib-model-actions">' +
            '<button class="btn btn-danger btn-sm img-lib-model-delete-btn" data-model="' + escapeHtml(m.name) + '">Delete</button>' +
          "</div></div>";
      }).join("");

      // Wire delete buttons
      list.querySelectorAll(".img-lib-model-delete-btn").forEach(function (btn) {
        btn.addEventListener("click", function () { imgLibDeleteModel(btn.dataset.model); });
      });
    } catch (_e) {
      list.innerHTML = '<div class="img-lib-empty-state">Error loading models</div>';
    }
  }

  async function imgLibDeleteModel(name) {
    if (!confirm('Delete model "' + name + '"? This will remove all model files.')) return;
    var base = appSettings.backendUrl || "";
    try {
      var resp = await fetch(base + "/api/image/models/" + encodeURIComponent(name), { method: "DELETE" });
      if (!resp.ok) {
        showToast("Failed to delete model", "error");
        return;
      }
      showToast("Model deleted", "success");
      imgLibLoadModels();
      // Refresh image panel model selector
      refreshImageModels();
    } catch (_e) {
      showToast("Error deleting model", "error");
    }
  }

  async function imgLibPullModel() {
    var sourceInput = $("#img-lib-model-source");
    var source = sourceInput.value.trim();
    if (!source) { showToast("Enter a model source", "error"); return; }

    var progressEl = $("#img-lib-pull-progress");
    var statusEl = $("#img-lib-pull-status");
    var fillEl = $("#img-lib-pull-fill");
    progressEl.classList.remove("hidden");
    statusEl.textContent = "Starting download...";
    fillEl.style.width = "0%";

    var base = appSettings.backendUrl || "";
    try {
      var resp = await fetch(base + "/api/image/models/pull", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: source }),
      });

      var reader = resp.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";

      while (true) {
        var result = await reader.read();
        if (result.done) break;
        buffer += decoder.decode(result.value, { stream: true });
        var lines = buffer.split("\n");
        buffer = lines.pop();

        for (var i = 0; i < lines.length; i++) {
          if (!lines[i].trim()) continue;
          try {
            var event = JSON.parse(lines[i]);
            if (event.status === "downloading") {
              var pct = event.progress || 0;
              statusEl.textContent = "Downloading... " + pct.toFixed(0) + "%";
              fillEl.style.width = pct + "%";
            } else if (event.status === "complete") {
              statusEl.textContent = "Complete!";
              fillEl.style.width = "100%";
              showToast("Model downloaded: " + (event.name || source), "success");
              sourceInput.value = "";
              imgLibLoadModels();
              refreshImageModels();
            } else if (event.status === "error") {
              statusEl.textContent = "Error: " + (event.error || "Unknown error");
              showToast("Download failed: " + (event.error || ""), "error");
            } else {
              statusEl.textContent = event.status || "Working...";
            }
          } catch (_e) { /* skip */ }
        }
      }
    } catch (err) {
      statusEl.textContent = "Error: " + err.message;
      showToast("Download failed", "error");
    }
    setTimeout(function () { progressEl.classList.add("hidden"); }, 3000);
  }

  // ---- Model Drag-and-Drop Upload ----

  function initModelDropzone() {
    var dropzone = $("#img-model-dropzone");
    var fileInput = $("#img-model-file-input");
    if (!dropzone || !fileInput) return;

    // Prevent default drag behaviors on the whole document
    ["dragenter", "dragover", "dragleave", "drop"].forEach(function (evt) {
      dropzone.addEventListener(evt, function (e) {
        e.preventDefault();
        e.stopPropagation();
      });
    });

    // Visual feedback
    ["dragenter", "dragover"].forEach(function (evt) {
      dropzone.addEventListener(evt, function () {
        dropzone.classList.add("drag-over");
      });
    });
    ["dragleave", "drop"].forEach(function (evt) {
      dropzone.addEventListener(evt, function () {
        dropzone.classList.remove("drag-over");
      });
    });

    // Handle drop
    dropzone.addEventListener("drop", function (e) {
      var files = e.dataTransfer.files;
      if (files.length > 0) handleModelFileUpload(files[0]);
    });

    // Handle file input (browse button)
    fileInput.addEventListener("change", function () {
      if (fileInput.files.length > 0) {
        handleModelFileUpload(fileInput.files[0]);
        fileInput.value = ""; // reset so same file can be re-selected
      }
    });

    // Click on dropzone opens file picker (unless clicking the browse link)
    dropzone.addEventListener("click", function (e) {
      if (e.target.tagName === "LABEL" || e.target.tagName === "INPUT") return;
      fileInput.click();
    });
  }

  async function handleModelFileUpload(file) {
    if (!file.name.endsWith(".safetensors")) {
      showToast("Only .safetensors files are supported", "error");
      return;
    }

    var progressEl = $("#img-model-upload-progress");
    var statusEl = $("#img-model-upload-status");
    var fillEl = $("#img-model-upload-fill");
    progressEl.classList.remove("hidden");

    var sizeMB = (file.size / 1024 / 1024).toFixed(0);
    statusEl.textContent = "Uploading " + file.name + " (" + sizeMB + " MB)...";
    fillEl.style.width = "0%";

    var base = appSettings.backendUrl || "";
    var formData = new FormData();
    formData.append("file", file);

    try {
      var xhr = new XMLHttpRequest();
      var uploadDone = new Promise(function (resolve, reject) {
        xhr.upload.addEventListener("progress", function (e) {
          if (e.lengthComputable) {
            var pct = Math.round((e.loaded / e.total) * 100);
            fillEl.style.width = pct + "%";
            statusEl.textContent = "Uploading " + file.name + "... " + pct + "%";
          }
        });
        xhr.addEventListener("load", function () {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(JSON.parse(xhr.responseText));
          } else {
            try {
              var err = JSON.parse(xhr.responseText);
              reject(new Error(err.detail || err.error || "Upload failed"));
            } catch (_e) {
              reject(new Error("Upload failed (HTTP " + xhr.status + ")"));
            }
          }
        });
        xhr.addEventListener("error", function () { reject(new Error("Network error")); });
        xhr.addEventListener("abort", function () { reject(new Error("Upload cancelled")); });
      });

      xhr.open("POST", base + "/api/image/models/upload");
      xhr.send(formData);

      var result = await uploadDone;
      fillEl.style.width = "100%";
      statusEl.textContent = "Imported: " + result.name;
      showToast("Model imported: " + result.name, "success");
      imgLibLoadModels();
      refreshImageModels();
    } catch (err) {
      statusEl.textContent = "Error: " + err.message;
      showToast("Upload failed: " + err.message, "error");
    }
    setTimeout(function () { progressEl.classList.add("hidden"); }, 3000);
  }

  // ---- Image Presets Management ----

  async function imgLibLoadPresets() {
    var list = $("#img-lib-presets-list");
    if (!list) return;
    var base = appSettings.backendUrl || "";
    try {
      var resp = await fetch(base + "/api/image/presets");
      if (!resp.ok) { list.innerHTML = '<div class="img-lib-empty-state">Could not load presets</div>'; return; }
      var presets = await resp.json();
      if (presets.length === 0) {
        list.innerHTML = '<div class="img-lib-empty-state">No presets. Create one below.</div>';
        return;
      }
      list.innerHTML = presets.map(function (p) {
        var isBuiltin = ["fantasy_rpg", "anime", "scifi", "horror", "realism"].indexOf(p.name) !== -1;
        var badge = isBuiltin ? '<span class="img-lib-preset-badge">Built-in</span>' : "";
        var deleteBtn = isBuiltin ? "" :
          '<button class="btn btn-danger btn-sm img-lib-preset-delete-btn" data-preset="' + escapeHtml(p.name) + '">Delete</button>';
        var tags = p.positive_tags ? p.positive_tags.substring(0, 80) + (p.positive_tags.length > 80 ? "..." : "") : "";
        return '<div class="img-lib-preset-card">' +
          '<div class="img-lib-preset-info">' +
            '<div class="img-lib-preset-name">' + escapeHtml(p.display_name || p.name) + "</div>" +
            (p.description ? '<div class="img-lib-preset-desc">' + escapeHtml(p.description) + "</div>" : "") +
            (tags ? '<div class="img-lib-preset-tags">' + escapeHtml(tags) + "</div>" : "") +
          "</div>" +
          '<div class="img-lib-model-meta">Steps: ' + p.steps + " &middot; CFG: " + p.cfg_scale + "</div>" +
          badge +
          '<div class="img-lib-model-actions">' + deleteBtn + "</div></div>";
      }).join("");

      list.querySelectorAll(".img-lib-preset-delete-btn").forEach(function (btn) {
        btn.addEventListener("click", function () { imgLibDeletePreset(btn.dataset.preset); });
      });
    } catch (_e) {
      list.innerHTML = '<div class="img-lib-empty-state">Error loading presets</div>';
    }
  }

  async function imgLibDeletePreset(name) {
    if (!confirm('Delete preset "' + name + '"?')) return;
    var base = appSettings.backendUrl || "";
    try {
      var resp = await fetch(base + "/api/image/presets/" + encodeURIComponent(name), { method: "DELETE" });
      if (!resp.ok) {
        showToast("Failed to delete preset", "error");
        return;
      }
      showToast("Preset deleted", "success");
      imgLibLoadPresets();
    } catch (_e) {
      showToast("Error deleting preset", "error");
    }
  }

  async function imgLibCreatePreset() {
    var name = ($("#img-lib-preset-name").value || "").trim().toLowerCase().replace(/\s+/g, "_");
    var displayName = ($("#img-lib-preset-display").value || "").trim();
    if (!name) { showToast("Preset name is required", "error"); return; }

    var base = appSettings.backendUrl || "";
    try {
      var resp = await fetch(base + "/api/image/presets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: name,
          display_name: displayName || name,
          description: ($("#img-lib-preset-desc").value || "").trim(),
          positive_tags: ($("#img-lib-preset-positive").value || "").trim(),
          negative_tags: ($("#img-lib-preset-negative").value || "").trim(),
          steps: parseInt($("#img-lib-preset-steps").value) || 20,
          cfg_scale: parseFloat($("#img-lib-preset-cfg").value) || 7.0,
        }),
      });
      if (!resp.ok) {
        showToast("Failed to create preset", "error");
        return;
      }
      showToast("Preset created: " + (displayName || name), "success");
      // Clear form
      $("#img-lib-preset-name").value = "";
      $("#img-lib-preset-display").value = "";
      $("#img-lib-preset-desc").value = "";
      $("#img-lib-preset-positive").value = "";
      $("#img-lib-preset-negative").value = "";
      $("#img-lib-preset-steps").value = "20";
      $("#img-lib-preset-cfg").value = "7.0";
      imgLibLoadPresets();
    } catch (_e) {
      showToast("Error creating preset", "error");
    }
  }

  function populateLibraryFilters() {
    // We populate with known values from loaded entries as they come in
    // Reset dropdowns
    var modelSel = $("#img-lib-model-filter");
    var presetSel = $("#img-lib-preset-filter");
    modelSel.innerHTML = '<option value="">All Models</option>';
    presetSel.innerHTML = '<option value="">All Presets</option>';
  }

  function updateFilterDropdowns() {
    var modelSel = $("#img-lib-model-filter");
    var presetSel = $("#img-lib-preset-filter");
    var models = new Set();
    var presets = new Set();
    imgLibState.entries.forEach(function (e) {
      if (e.model) models.add(e.model);
      if (e.preset) presets.add(e.preset);
    });
    var currentModel = modelSel.value;
    var currentPreset = presetSel.value;
    modelSel.innerHTML = '<option value="">All Models</option>';
    models.forEach(function (m) {
      var opt = document.createElement("option");
      opt.value = m; opt.textContent = m;
      modelSel.appendChild(opt);
    });
    presetSel.innerHTML = '<option value="">All Presets</option>';
    presets.forEach(function (p) {
      var opt = document.createElement("option");
      opt.value = p; opt.textContent = p;
      presetSel.appendChild(opt);
    });
    modelSel.value = currentModel;
    presetSel.value = currentPreset;
  }

  function imgLibResetAndLoad() {
    imgLibState.offset = 0;
    imgLibState.entries = [];
    var grid = $("#img-lib-grid");
    grid.innerHTML = "";
    // Show skeletons
    for (var i = 0; i < 12; i++) {
      var skel = document.createElement("div");
      skel.className = "img-lib-skeleton";
      grid.appendChild(skel);
    }
    $("#img-lib-empty").classList.remove("visible");
    imgLibFetchPage(false);
  }

  async function imgLibFetchPage(append) {
    if (imgLibState.loading) return;
    imgLibState.loading = true;

    var base = appSettings.backendUrl || "";
    var q = ($("#img-lib-search") || {}).value || "";
    var model = ($("#img-lib-model-filter") || {}).value || "";
    var preset = ($("#img-lib-preset-filter") || {}).value || "";
    var sort = ($("#img-lib-sort") || {}).value || "newest";
    var limit = 48;

    var url = base + "/api/image/history?limit=" + limit +
      "&offset=" + imgLibState.offset +
      "&q=" + encodeURIComponent(q) +
      "&model=" + encodeURIComponent(model) +
      "&preset=" + encodeURIComponent(preset) +
      "&sort=" + encodeURIComponent(sort);

    try {
      var resp = await fetch(url);
      if (!resp.ok) {
        imgLibState.loading = false;
        return;
      }
      var data = await resp.json();
      var entries = data.entries || [];
      imgLibState.total = data.total || 0;

      // Add base URL to entry urls
      entries.forEach(function (e) {
        e.url = base + (e.url || "/api/image/" + e.image_id);
      });

      if (!append) {
        var grid = $("#img-lib-grid");
        grid.innerHTML = "";
        imgLibState.entries = entries;
      } else {
        imgLibState.entries = imgLibState.entries.concat(entries);
      }

      imgLibRenderEntries(entries, append);

      // Update count
      $("#img-lib-count").textContent = imgLibState.total + " image" + (imgLibState.total !== 1 ? "s" : "");

      // Toggle load more
      var loaded = imgLibState.entries.length;
      var loadMore = $("#img-lib-load-more");
      if (loaded < imgLibState.total) {
        loadMore.classList.remove("hidden");
      } else {
        loadMore.classList.add("hidden");
      }

      // Toggle empty state
      var empty = $("#img-lib-empty");
      if (imgLibState.total === 0) {
        empty.classList.add("visible");
      } else {
        empty.classList.remove("visible");
      }

      updateFilterDropdowns();
    } catch (_e) {
      // silently fail
    }
    imgLibState.loading = false;
  }

  function imgLibRenderEntries(entries, append) {
    var grid = $("#img-lib-grid");
    if (!append) grid.innerHTML = "";

    for (var i = 0; i < entries.length; i++) {
      if (imgLibState.view === "list") {
        grid.appendChild(imgLibBuildRow(entries[i]));
      } else {
        grid.appendChild(imgLibBuildCard(entries[i]));
      }
    }
  }

  function imgLibBuildCard(entry) {
    var card = document.createElement("div");
    card.className = "img-lib-card";
    card.dataset.imageId = entry.image_id;

    if (imgLibState.selectedIds.has(entry.image_id)) {
      card.classList.add("selected");
    }

    var img = document.createElement("img");
    img.src = entry.url;
    img.alt = entry.prompt || "";
    img.loading = "lazy";
    card.appendChild(img);

    var overlay = document.createElement("div");
    overlay.className = "img-lib-card-overlay";

    var promptEl = document.createElement("div");
    promptEl.className = "img-lib-card-prompt";
    promptEl.textContent = entry.prompt || "";
    overlay.appendChild(promptEl);

    var actions = document.createElement("div");
    actions.className = "img-lib-card-actions";
    actions.innerHTML =
      '<button class="img-lib-card-action" data-action="download">DL</button>' +
      '<button class="img-lib-card-action" data-action="delete">Del</button>';
    overlay.appendChild(actions);
    card.appendChild(overlay);

    var check = document.createElement("div");
    check.className = "img-lib-card-check";
    if (imgLibState.selectedIds.has(entry.image_id)) check.innerHTML = "&#10003;";
    card.appendChild(check);

    return card;
  }

  function imgLibBuildRow(entry) {
    var row = document.createElement("div");
    row.className = "img-lib-row";
    row.dataset.imageId = entry.image_id;

    if (imgLibState.selectedIds.has(entry.image_id)) {
      row.classList.add("selected");
    }

    var thumb = document.createElement("img");
    thumb.className = "img-lib-row-thumb";
    thumb.src = entry.url;
    thumb.alt = entry.prompt || "";
    thumb.loading = "lazy";
    row.appendChild(thumb);

    var info = document.createElement("div");
    info.className = "img-lib-row-info";
    var p = document.createElement("div");
    p.className = "img-lib-row-prompt";
    p.textContent = entry.prompt || "";
    info.appendChild(p);

    var meta = document.createElement("div");
    meta.className = "img-lib-row-meta";
    if (entry.model) meta.innerHTML += '<span class="img-lib-row-chip">' + escapeHtml(entry.model) + "</span>";
    meta.innerHTML += '<span class="img-lib-row-chip">' + entry.width + "x" + entry.height + "</span>";
    if (entry.seed != null && entry.seed !== -1) meta.innerHTML += '<span class="img-lib-row-chip">seed: ' + entry.seed + "</span>";
    info.appendChild(meta);
    row.appendChild(info);

    var actions = document.createElement("div");
    actions.className = "img-lib-row-actions";
    actions.innerHTML =
      '<button class="img-lib-card-action" data-action="download">DL</button>' +
      '<button class="img-lib-card-action" data-action="delete">Del</button>';
    row.appendChild(actions);

    var check = document.createElement("div");
    check.className = "img-lib-row-check";
    if (imgLibState.selectedIds.has(entry.image_id)) check.innerHTML = "&#10003;";
    row.appendChild(check);

    return row;
  }

  function escapeHtml(str) {
    var d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
  }

  function imgLibLoadMore() {
    imgLibState.offset += 48;
    imgLibFetchPage(true);
  }

  function imgLibSelectEntry(entry) {
    imgLibState.currentEntry = entry;
    var detail = $("#img-lib-detail");
    var img = $("#img-lib-detail-img");
    var metaEl = $("#img-lib-detail-meta");

    img.src = entry.url;
    metaEl.innerHTML = "";

    var fields = [
      ["Prompt", entry.prompt || ""],
      ["Negative", entry.negative_prompt || ""],
      ["Model", entry.model || ""],
      ["Preset", entry.preset || ""],
      ["Seed", entry.seed != null ? String(entry.seed) : ""],
      ["Steps", entry.steps ? String(entry.steps) : ""],
      ["CFG", entry.cfg_scale ? String(entry.cfg_scale) : ""],
      ["Size", entry.width + " x " + entry.height],
      ["Date", entry.created_at || ""],
    ];

    if (entry.loras && entry.loras.length > 0) {
      var loraStr = entry.loras.map(function (l) { return l.name + " (" + l.weight + ")"; }).join(", ");
      fields.push(["LoRAs", loraStr]);
    }

    for (var i = 0; i < fields.length; i++) {
      if (!fields[i][1]) continue;
      var label = document.createElement("span");
      label.className = "img-lib-meta-label";
      label.textContent = fields[i][0];
      metaEl.appendChild(label);
      var val = document.createElement("span");
      val.className = "img-lib-meta-value";
      val.textContent = fields[i][1];
      metaEl.appendChild(val);
    }

    detail.classList.remove("hidden");
  }

  function closeLibraryDetail() {
    var detail = $("#img-lib-detail");
    if (detail) detail.classList.add("hidden");
    imgLibState.currentEntry = null;
  }

  function setLibraryView(mode) {
    imgLibState.view = mode;
    var grid = $("#img-lib-grid");
    var gridBtn = $("#img-lib-view-grid");
    var listBtn = $("#img-lib-view-list");

    if (mode === "list") {
      grid.classList.add("list-view");
      listBtn.classList.add("active");
      gridBtn.classList.remove("active");
    } else {
      grid.classList.remove("list-view");
      gridBtn.classList.add("active");
      listBtn.classList.remove("active");
    }

    // Re-render with current entries
    imgLibRenderEntries(imgLibState.entries, false);
  }

  function toggleBulkMode() {
    imgLibState.bulkMode = !imgLibState.bulkMode;
    var grid = $("#img-lib-grid");
    var selectBtn = $("#img-lib-select-btn");
    var bulkDeleteBtn = $("#img-lib-bulk-delete-btn");

    if (imgLibState.bulkMode) {
      grid.classList.add("bulk-mode");
      selectBtn.textContent = "Cancel";
      bulkDeleteBtn.classList.remove("hidden");
    } else {
      grid.classList.remove("bulk-mode");
      selectBtn.textContent = "Select";
      bulkDeleteBtn.classList.add("hidden");
      imgLibState.selectedIds.clear();
      // Remove selected state from all cards
      var selected = grid.querySelectorAll(".selected");
      for (var i = 0; i < selected.length; i++) {
        selected[i].classList.remove("selected");
        var chk = selected[i].querySelector(".img-lib-card-check, .img-lib-row-check");
        if (chk) chk.innerHTML = "";
      }
    }
    updateBulkDeleteLabel();
  }

  function updateBulkDeleteLabel() {
    var btn = $("#img-lib-bulk-delete-btn");
    var count = imgLibState.selectedIds.size;
    btn.textContent = count > 0 ? "Delete Selected (" + count + ")" : "Delete Selected";
  }

  async function handleLibraryDelete(imageId) {
    var id = imageId || (imgLibState.currentEntry && imgLibState.currentEntry.image_id);
    if (!id) return;
    if (!confirm("Delete this image permanently?")) return;

    var base = appSettings.backendUrl || "";
    try {
      var resp = await fetch(base + "/api/image/" + id, { method: "DELETE" });
      if (!resp.ok) {
        showToast("Failed to delete image", "error");
        return;
      }

      // Remove from state
      imgLibState.entries = imgLibState.entries.filter(function (e) { return e.image_id !== id; });
      imgLibState.total = Math.max(0, imgLibState.total - 1);
      imgLibState.selectedIds.delete(id);

      // Remove from DOM
      var card = document.querySelector('[data-image-id="' + id + '"]');
      if (card) card.remove();

      // Update count
      $("#img-lib-count").textContent = imgLibState.total + " image" + (imgLibState.total !== 1 ? "s" : "");

      // Close detail if showing this image
      if (imgLibState.currentEntry && imgLibState.currentEntry.image_id === id) {
        closeLibraryDetail();
      }

      if (imgLibState.total === 0) {
        $("#img-lib-empty").classList.add("visible");
      }

      // Refresh sidebar gallery
      refreshImageGallery();
    } catch (_e) {
      showToast("Error deleting image", "error");
    }
  }

  async function handleBulkDelete() {
    var ids = Array.from(imgLibState.selectedIds);
    if (ids.length === 0) return;
    if (!confirm("Delete " + ids.length + " image" + (ids.length > 1 ? "s" : "") + " permanently?")) return;

    var base = appSettings.backendUrl || "";
    try {
      var resp = await fetch(base + "/api/image/batch", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_ids: ids }),
      });
      if (!resp.ok) {
        showToast("Failed to delete images", "error");
        return;
      }
      var result = await resp.json();
      var deleted = result.deleted || [];

      // Remove deleted from state
      deleted.forEach(function (delId) {
        imgLibState.entries = imgLibState.entries.filter(function (e) { return e.image_id !== delId; });
        imgLibState.selectedIds.delete(delId);
        var card = document.querySelector('[data-image-id="' + delId + '"]');
        if (card) card.remove();
      });

      imgLibState.total = Math.max(0, imgLibState.total - deleted.length);
      $("#img-lib-count").textContent = imgLibState.total + " image" + (imgLibState.total !== 1 ? "s" : "");
      updateBulkDeleteLabel();

      if (imgLibState.currentEntry && deleted.indexOf(imgLibState.currentEntry.image_id) !== -1) {
        closeLibraryDetail();
      }

      if (imgLibState.total === 0) {
        $("#img-lib-empty").classList.add("visible");
      }

      // Refresh sidebar gallery
      refreshImageGallery();
    } catch (_e) {
      showToast("Error deleting images", "error");
    }
  }

  function handleLibraryDownload() {
    if (!imgLibState.currentEntry) return;
    var a = document.createElement("a");
    a.href = imgLibState.currentEntry.url;
    a.download = imgLibState.currentEntry.image_id + ".png";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  function handleLibraryCopySeed() {
    if (!imgLibState.currentEntry) return;
    var seed = String(imgLibState.currentEntry.seed);
    navigator.clipboard.writeText(seed).then(function () {
      var btn = $("#img-lib-copy-seed");
      var orig = btn.textContent;
      btn.textContent = "Copied!";
      setTimeout(function () { btn.textContent = orig; }, 1500);
    });
  }

  function handleLibraryReusePrompt() {
    if (!imgLibState.currentEntry) return;
    var entry = imgLibState.currentEntry;

    var promptEl = $("#img-prompt");
    var negEl = $("#img-negative");
    var presetEl = $("#img-preset");
    var seedEl = $("#img-seed");
    var widthEl = $("#img-width");
    var heightEl = $("#img-height");
    var stepsEl = $("#img-steps");
    var cfgEl = $("#img-cfg");

    if (promptEl) promptEl.value = entry.prompt || "";
    if (negEl) negEl.value = entry.negative_prompt || "";
    if (presetEl) presetEl.value = entry.preset || "";
    if (seedEl) seedEl.value = -1;  // always randomize on reuse
    if (widthEl && entry.width) widthEl.value = entry.width;
    if (heightEl && entry.height) heightEl.value = entry.height;
    if (stepsEl && entry.steps) stepsEl.value = entry.steps;
    if (cfgEl && entry.cfg_scale) cfgEl.value = entry.cfg_scale;

    renderResolutionPresets();
    closeImageLibrary();

    // Open image panel if hidden
    var panel = $("#image-panel");
    if (panel && panel.classList.contains("hidden")) {
      panel.classList.remove("hidden");
    }
  }

  // ---- Start ----

  init();

  // Initialize image panel after DOM is ready
  initImagePanel();
  initImageLibrary();

  // Reconnect to any active model downloads (survives page reload)
  (async function reconnectPullTasks() {
    try {
      var base = appSettings.backendUrl || "";
      var resp = await fetch(base + "/api/image/models/pull");
      if (!resp.ok) return;
      var tasks = await resp.json();
      if (!tasks || !tasks.length) return;

      for (var i = 0; i < tasks.length; i++) {
        var t = tasks[i];
        if (t.status !== "running") continue;
        // Show progress in the pull area
        var progressArea = $("#img-pull-progress");
        var fill = $("#img-pull-fill");
        var statusEl = $("#img-pull-status");
        if (progressArea) progressArea.classList.remove("hidden");
        if (statusEl) statusEl.textContent = "Resuming download...";
        if (fill) fill.style.width = (t.percent || 0) + "%";

        pollPullTask(t.task_id,
          function (data) {
            if (data.percent !== undefined) {
              if (fill) fill.style.width = data.percent + "%";
              var pctText = Math.round(data.percent) + "%";
              if (data.files_done && data.files_total) pctText += " (" + data.files_done + "/" + data.files_total + " files)";
              if (statusEl) statusEl.textContent = pctText;
            }
          },
          function () {
            if (fill) fill.style.width = "100%";
            if (statusEl) statusEl.textContent = "Complete!";
            refreshImageModels();
            refreshImageCatalog();
            setTimeout(function () { if (progressArea) progressArea.classList.add("hidden"); }, 3000);
          },
          function (errMsg) {
            if (statusEl) statusEl.textContent = "Error: " + errMsg;
            setTimeout(function () { if (progressArea) progressArea.classList.add("hidden"); }, 4000);
          }
        );
        break; // Only show one active download in the pull progress area
      }
    } catch (_) { /* ignore */ }
  })();

  // Restore sidebar and panel states from settings
  if (dom.sidebar) {
    if (appSettings.sidebarOpen) {
      dom.sidebar.classList.remove("collapsed");
    } else {
      dom.sidebar.classList.add("collapsed");
    }
  }
  if (appSettings.imagePanelOpen) {
    var imgPanel = $("#image-panel");
    if (imgPanel) {
      imgPanel.classList.remove("hidden");
      fetchImageHardware();
      refreshImageModels();
      refreshImageGallery();
      refreshImageCatalog();
      fetchImageSamplers();
      renderResolutionPresets();
    }
  }
})();
