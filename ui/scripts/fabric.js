// Fabric tab — operator surface for cross-instance peer coordination.
//
// Default-off invariant: the tab renders cleanly when fabric_enabled
// is False (the most common case) by showing the "fabric is off"
// state. Every string that touches user-provided data goes through
// escapeHtml() — operator-supplied addresses, peer hostnames, and
// remote-advertised capability names all originate from outside this
// node and must never be trusted in a template literal.
//
// Phase 4 ships:
//   - identity card (this node's fingerprint + copy-to-clipboard)
//   - enable toggle (flips fabric_enabled in settings store)
//   - paired peer list (status, fingerprint, capability count, unpair)
//   - pair flow (paste a remote node's URL + fingerprint to add)
//
// Phase 4.y will add:
//   - live capability matrix
//   - routing rule editor
//   - per-session pin badge in chat header
//   - peer health/latency graphs

import { escapeHtml, showToast } from './app.js';

// Phase 8 — curated peer-icon grid. Mirrors augmentum/fabric/icons.py
// (PEER_ICONS tuple). Keep these in sync; the order shown to operators
// is whatever this array uses. "Other…" at the end is an escape hatch
// that lets the operator paste any single emoji.
const PEER_ICON_GRID = [
  '🏎', '🚀', '⚡', '🐢', '🐌',
  '🦊', '🐻', '🐉', '🦉', '🐙',
  '🦅', '🐺', '🐳', '🌋', '⛰',
  '🌊', '❄', '🔥', '🌟', '🛰',
];
const DEFAULT_PEER_ICON = '🔗';

let _fabricEnabled = false;
let _fabricBound = false;
// Currently-selected pair-form icon. Defaults to empty string ("no icon
// chosen"); the backend stores '' which the UI maps to DEFAULT_PEER_ICON
// at render time.
let _pickedPeerIcon = '';
// Result of the most recent LAN sweep. Null until the operator has
// clicked Scan at least once; resets to null on tab re-open. The shape
// matches the /api/fabric/discover response (peers / self_seen /
// already_paired / errors / hosts_probed / duration_s).
let _lastDiscovery = null;
let _isScanning = false;

// ── Public entry point (called from models.js when the tab opens) ──

export async function renderFabricTab() {
  const root = document.getElementById('mm-fabric-root');
  if (!root) return;

  // Lazy bind once -- the tab DOM is static; only the inner content
  // re-renders on state refresh.
  if (!_fabricBound) {
    _bindEvents();
    _fabricBound = true;
  }

  await refreshFabricState();
}

// ── State refresh ──────────────────────────────────────────────────

async function refreshFabricState() {
  const root = document.getElementById('mm-fabric-root');
  if (!root) return;

  let statusData = null;
  let peersData = null;
  let capsData = null;
  try {
    const [statusResp, peersResp, capsResp] = await Promise.all([
      fetch('/api/fabric/status'),
      fetch('/api/fabric/peers'),
      fetch('/api/fabric/capabilities'),
    ]);
    if (statusResp.ok) statusData = await statusResp.json();
    if (peersResp.ok) peersData = await peersResp.json();
    if (capsResp.ok) capsData = await capsResp.json();
  } catch (err) {
    root.innerHTML = `
      <div class="mm-section">
        <div class="mm-section-title">Fabric is unavailable</div>
        <div class="mm-section-copy">
          The fabric endpoints could not be reached: ${escapeHtml(String(err && err.message || err))}.
        </div>
      </div>
    `;
    return;
  }

  if (!statusData) {
    root.innerHTML = `
      <div class="mm-section">
        <div class="mm-section-title">Fabric is unavailable</div>
        <div class="mm-section-copy">
          The /api/fabric/status endpoint returned an error. You may need to be signed in as an admin.
        </div>
      </div>
    `;
    return;
  }

  _fabricEnabled = Boolean(statusData.enabled);
  // Stash the local-fabric-icon (operator's self-label) so matrix view
  // can use it in the "This node" column header without an extra fetch.
  window.__augmentumLocalFabricIcon = (statusData.this_node && statusData.this_node.icon) || '';

  root.innerHTML = `
    ${_renderToggleSection()}
    ${_renderIdentitySection(statusData)}
    ${_renderPeerListSection(peersData)}
    ${_renderCapabilityMatrixSection(capsData, peersData)}
    ${_renderDiscoverSection()}
    ${_renderPairSection()}
  `;

  _bindEvents();
}

// ── Section renderers (return HTML strings) ────────────────────────

function _renderToggleSection() {
  // The toggle persists via the standard tool-settings sync flow.
  // We render the checkbox based on the LIVE backend status (read
  // from /api/fabric/status) rather than the cached settings dict
  // so a fresh tab open shows the truthful state.
  return `
    <div class="mm-section">
      <div class="mm-section-header mm-section-header-stack">
        <div>
          <div class="mm-section-title">Fabric layer</div>
          <div class="mm-section-copy">
            Cross-instance peer coordination -- pair another Augmentum
            to share LLM, image, and knowledge capabilities across your
            devices. Default off. Turning this on instantiates the
            coordinator + signing identity; no traffic flows until you
            also pair a peer.
          </div>
        </div>
      </div>
      <label class="mm-fabric-toggle">
        <input type="checkbox" id="mm-fabric-enabled-toggle" ${_fabricEnabled ? 'checked' : ''}>
        <span>Enable the fabric layer</span>
      </label>
      <div class="mm-fabric-hint">
        Changing this requires an Augmentum restart to take effect.
      </div>
    </div>
  `;
}

function _renderIdentitySection(statusData) {
  if (!_fabricEnabled || !statusData.this_node) {
    return `
      <div class="mm-section">
        <div class="mm-section-title">This node</div>
        <div class="mm-section-copy">
          Fabric is disabled. Enable it above + restart to generate
          this node's ed25519 identity.
        </div>
      </div>
    `;
  }

  const nodeId = statusData.this_node.node_id || '';
  const fingerprint = statusData.this_node.fingerprint || '';
  const localIcon = (statusData.this_node && statusData.this_node.icon) || '';

  return `
    <div class="mm-section">
      <div class="mm-section-header mm-section-header-stack">
        <div>
          <div class="mm-section-title">This node</div>
          <div class="mm-section-copy">
            Share this fingerprint with a peer operator to pair. The
            ed25519 keypair was generated at first fabric startup and
            is encrypted at rest in this instance's settings store.
          </div>
        </div>
      </div>
      <div class="mm-fabric-identity">
        <div class="mm-fabric-identity-row">
          <span class="mm-fabric-identity-label">Node ID</span>
          <span class="mm-fabric-identity-value mm-fabric-mono">${escapeHtml(nodeId)}</span>
        </div>
        <div class="mm-fabric-identity-row">
          <span class="mm-fabric-identity-label">Fingerprint</span>
          <span class="mm-fabric-identity-value mm-fabric-mono">${escapeHtml(fingerprint)}</span>
          <button class="btn btn-sm mm-fabric-copy" data-copy="${escapeHtml(fingerprint)}">
            Copy
          </button>
        </div>
        <div class="mm-fabric-identity-row">
          <span class="mm-fabric-identity-label">Icon</span>
          <span class="mm-fabric-identity-value">
            <span class="mm-fabric-peer-icon" id="mm-fabric-local-icon-preview">${escapeHtml(localIcon || DEFAULT_PEER_ICON)}</span>
            <input type="text" class="field-input mm-fabric-icon-other"
                   id="mm-fabric-local-icon-input" maxlength="8"
                   placeholder="🏎"
                   value="${escapeHtml(localIcon)}"
                   style="margin-left:8px;">
          </span>
        </div>
      </div>
      <div class="mm-fabric-hint">
        Your own box's icon — surfaces in the capability matrix column
        header and on chat turns this node serves to your peers.
      </div>
    </div>
  `;
}

function _renderPeerListSection(peersData) {
  if (!_fabricEnabled) return '';

  const peers = (peersData && peersData.peers) || [];

  if (peers.length === 0) {
    return `
      <div class="mm-section">
        <div class="mm-section-title">Paired peers (0)</div>
        <div class="mm-section-copy">
          No peers paired yet. Use the section below to add one.
        </div>
      </div>
    `;
  }

  const rows = peers.map(_renderPeerRow).join('');
  return `
    <div class="mm-section">
      <div class="mm-section-header mm-section-header-stack">
        <div>
          <div class="mm-section-title">Paired peers (${peers.length})</div>
          <div class="mm-section-copy">
            Connection status is reported as of the last fabric heartbeat
            (sent every ~5s when fabric is enabled).
          </div>
        </div>
      </div>
      <div class="mm-fabric-peer-list">
        ${rows}
      </div>
    </div>
  `;
}

function _renderPeerRow(peer) {
  const hostname = peer.hostname || peer.node_id || '(unknown)';
  const icon = peer.icon || DEFAULT_PEER_ICON;
  const statusPill = peer.connected
    ? `<span class="mm-fabric-pill mm-fabric-pill-on">Connected</span>`
    : `<span class="mm-fabric-pill mm-fabric-pill-off">Offline</span>`;
  const role = peer.role === 'primary' ? 'Primary' : 'Peer';
  const capBlurb = peer.capability_count > 0
    ? `${peer.capability_count} capabilities advertised`
    : peer.connected ? 'No capabilities advertised yet' : 'No capabilities (offline)';
  // Show "Last seen" only when offline + a stamp exists; for connected
  // peers it would always read "just now" which adds clutter without
  // signal.
  const lastSeenLine = (!peer.connected && peer.last_seen_at)
    ? `<div class="mm-fabric-peer-meta">
         <span><strong>Last seen</strong> ${escapeHtml(peer.last_seen_at)}</span>
       </div>`
    : '';

  return `
    <div class="mm-fabric-peer" data-node-id="${escapeHtml(peer.node_id)}">
      <div class="mm-fabric-peer-header">
        <div class="mm-fabric-peer-name">
          <span class="mm-fabric-peer-icon" title="${escapeHtml(hostname)}">${escapeHtml(icon)}</span>
          ${escapeHtml(hostname)}
        </div>
        ${statusPill}
        <button class="btn btn-sm mm-fabric-unpair" data-node-id="${escapeHtml(peer.node_id)}">
          Unpair
        </button>
      </div>
      <div class="mm-fabric-peer-meta">
        <span><strong>Role</strong> ${escapeHtml(role)}</span>
        <span><strong>Addr</strong> ${escapeHtml(peer.addr || '(none)')}</span>
        <span><strong>Tier</strong> ${escapeHtml(peer.tier || 'local')}</span>
      </div>
      <div class="mm-fabric-peer-meta">
        <span class="mm-fabric-mono">${escapeHtml(peer.fingerprint || '')}</span>
      </div>
      ${lastSeenLine}
      <div class="mm-fabric-peer-caps">
        ${escapeHtml(capBlurb)}
      </div>
    </div>
  `;
}

// ── Capability matrix ─────────────────────────────────────────────
//
// Shape: one subsection per kind (LLM, image, knowledge). Within each
// subsection a grid where rows are the unique model_id / pack_id seen
// across the fabric, and columns are "This node" + each connected
// peer. Cells render a status badge (loaded / ready / —) plus the
// most useful per-kind detail (free_slots for LLM, max_resolution
// for image, chunk_count for knowledge).
//
// Read-only — operator gets a fabric-wide inventory at a glance, no
// rules to manage yet (that's a follow-up).

function _renderCapabilityMatrixSection(capsData, peersData) {
  if (!_fabricEnabled) return '';
  if (!capsData) {
    return `
      <div class="mm-section">
        <div class="mm-section-title">Capability matrix</div>
        <div class="mm-section-copy">
          Capability inventory is unavailable. The fabric coordinator
          may still be initialising.
        </div>
      </div>
    `;
  }

  const localCaps = Array.isArray(capsData.local) ? capsData.local : [];
  const peerCaps = (capsData.peers && typeof capsData.peers === 'object')
    ? capsData.peers
    : {};

  // Build the column list: "This node" first, then each peer that has
  // at least one capability. Pull hostnames + icons from /peers data
  // when available; fall back to a truncated node_id + DEFAULT_PEER_ICON.
  const peerMetaByNodeId = new Map();
  if (peersData && Array.isArray(peersData.peers)) {
    for (const p of peersData.peers) {
      if (p && p.node_id) {
        peerMetaByNodeId.set(p.node_id, {
          hostname: p.hostname || '',
          icon: p.icon || DEFAULT_PEER_ICON,
        });
      }
    }
  }
  const peerColumns = Object.keys(peerCaps).map((nodeId) => {
    const meta = peerMetaByNodeId.get(nodeId) || {};
    return {
      nodeId,
      label: meta.hostname || nodeId.slice(0, 12),
      icon: meta.icon || DEFAULT_PEER_ICON,
    };
  });
  // Local-node column uses the operator-chosen local_fabric_icon if set.
  const localIcon = (window.__augmentumLocalFabricIcon) || '';
  const allColumns = [
    { nodeId: '__local__', label: 'This node', icon: localIcon },
    ...peerColumns,
  ];

  // If everything is empty, render an empty-state instead of an
  // empty matrix.
  const haveAnything = localCaps.length > 0 || peerColumns.length > 0;
  if (!haveAnything) {
    return `
      <div class="mm-section">
        <div class="mm-section-title">Capability matrix</div>
        <div class="mm-section-copy">
          No capabilities advertised yet. Capabilities populate as the
          local engine + paired peers report what they can do
          (LLM models, image pipelines, knowledge packs).
        </div>
      </div>
    `;
  }

  const llmGrid = _renderKindGrid('llm.inference', allColumns, localCaps, peerCaps);
  const imageGrid = _renderKindGrid('image.generation', allColumns, localCaps, peerCaps);
  const ttsGrid = _renderKindGrid('tts.synthesize', allColumns, localCaps, peerCaps);
  const sttGrid = _renderKindGrid('stt.transcribe', allColumns, localCaps, peerCaps);
  const knowledgeGrid = _renderKindGrid('knowledge.search', allColumns, localCaps, peerCaps);
  const castGrid = _renderKindGrid('cast.render', allColumns, localCaps, peerCaps);

  return `
    <div class="mm-section">
      <div class="mm-section-header mm-section-header-stack">
        <div>
          <div class="mm-section-title">Capability matrix</div>
          <div class="mm-section-copy">
            What every reachable node in your fabric is currently
            offering. Updates each refresh; capabilities are reported
            on peer heartbeats.
          </div>
        </div>
      </div>
      ${llmGrid}
      ${imageGrid}
      ${ttsGrid}
      ${sttGrid}
      ${knowledgeGrid}
      ${castGrid}
    </div>
  `;
}

function _renderKindGrid(kind, columns, localCaps, peerCaps) {
  const title = ({
    'llm.inference': 'LLM inference',
    'image.generation': 'Image generation',
    'tts.synthesize': 'Text-to-speech',
    'stt.transcribe': 'Speech-to-text',
    'knowledge.search': 'Knowledge packs',
    'cast.render': 'Cast render',
  })[kind] || kind;

  // Each kind has its own natural row identifier:
  //  - llm.inference, image.generation → model_id
  //  - knowledge.search → pack_id
  //  - tts.synthesize, stt.transcribe → provider_id (per-engine row)
  //  - cast.render → one row per node (no per-model key; key by kind itself)
  let rowKeyField;
  if (kind === 'knowledge.search') rowKeyField = 'pack_id';
  else if (kind === 'tts.synthesize' || kind === 'stt.transcribe') rowKeyField = 'provider_id';
  else if (kind === 'cast.render') rowKeyField = 'kind';
  else rowKeyField = 'model_id';
  const rowKeys = new Set();
  const _addRowKey = (caps) => {
    for (const c of caps) {
      if (c && c.kind === kind) {
        const k = c[rowKeyField];
        if (k) rowKeys.add(k);
      }
    }
  };
  _addRowKey(localCaps);
  for (const nodeId of Object.keys(peerCaps)) _addRowKey(peerCaps[nodeId]);

  if (rowKeys.size === 0) {
    return `
      <div class="mm-fabric-matrix-empty">
        <strong>${escapeHtml(title)}</strong> · no capabilities of this kind
      </div>
    `;
  }

  const sortedRowKeys = [...rowKeys].sort();
  const headerCells = columns
    .map((c) => {
      const iconHtml = c.icon
        ? `<span class="mm-fabric-peer-icon" title="${escapeHtml(c.label)}">${escapeHtml(c.icon)}</span>`
        : '';
      return `<th>${iconHtml}${escapeHtml(c.label)}</th>`;
    })
    .join('');

  const rows = sortedRowKeys.map((rowKey) => {
    const cells = columns.map((col) => {
      const sourceCaps = col.nodeId === '__local__'
        ? localCaps
        : (peerCaps[col.nodeId] || []);
      const match = sourceCaps.find(
        (c) => c && c.kind === kind && c[rowKeyField] === rowKey,
      );
      return `<td>${_renderCapCell(kind, match)}</td>`;
    }).join('');
    return `
      <tr>
        <td class="mm-fabric-matrix-rowkey">${escapeHtml(rowKey)}</td>
        ${cells}
      </tr>
    `;
  }).join('');

  return `
    <div class="mm-fabric-matrix-kind">
      <div class="mm-fabric-matrix-title">${escapeHtml(title)}</div>
      <table class="mm-fabric-matrix">
        <thead>
          <tr>
            <th class="mm-fabric-matrix-rowkey-h">${escapeHtml(_rowKeyLabel(kind))}</th>
            ${headerCells}
          </tr>
        </thead>
        <tbody>
          ${rows}
        </tbody>
      </table>
    </div>
  `;
}

function _rowKeyLabel(kind) {
  if (kind === 'knowledge.search') return 'Pack';
  if (kind === 'tts.synthesize' || kind === 'stt.transcribe') return 'Engine';
  if (kind === 'cast.render') return 'Surface';
  return 'Model';
}

function _renderCapCell(kind, cap) {
  if (!cap) {
    return `<span class="mm-fabric-matrix-cell-missing">—</span>`;
  }
  if (kind === 'llm.inference') {
    const status = cap.loaded ? 'loaded' : 'ready';
    const pillCls = cap.loaded ? 'mm-fabric-pill-on' : 'mm-fabric-pill-off';
    const slots = (typeof cap.free_slots === 'number' && cap.free_slots > 0)
      ? ` · ${cap.free_slots} slots`
      : '';
    const ctx = cap.ctx_max ? ` · ${_formatCtx(cap.ctx_max)} ctx` : '';
    return `
      <span class="mm-fabric-pill ${pillCls}">${escapeHtml(status)}</span>
      <span class="mm-fabric-matrix-cell-detail">${escapeHtml(slots + ctx)}</span>
    `;
  }
  if (kind === 'image.generation') {
    const status = cap.loaded ? 'loaded' : 'ready';
    const pillCls = cap.loaded ? 'mm-fabric-pill-on' : 'mm-fabric-pill-off';
    const res = cap.max_resolution ? ` · ${cap.max_resolution}` : '';
    return `
      <span class="mm-fabric-pill ${pillCls}">${escapeHtml(status)}</span>
      <span class="mm-fabric-matrix-cell-detail">${escapeHtml(res)}</span>
    `;
  }
  if (kind === 'knowledge.search') {
    const chunks = (typeof cap.chunk_count === 'number' && cap.chunk_count > 0)
      ? `${_formatCount(cap.chunk_count)} chunks`
      : 'present';
    return `
      <span class="mm-fabric-pill mm-fabric-pill-on">✓</span>
      <span class="mm-fabric-matrix-cell-detail">${escapeHtml(chunks)}</span>
    `;
  }
  if (kind === 'tts.synthesize') {
    const voiceCount = Array.isArray(cap.voices) ? cap.voices.length : 0;
    const detail = [
      cap.in_process ? 'built-in' : 'sidecar',
      voiceCount > 0 ? `${voiceCount} voices` : '',
      Array.isArray(cap.languages) && cap.languages.length > 0
        ? cap.languages.slice(0, 3).join('/')
        : '',
    ].filter(Boolean).join(' · ');
    return `
      <span class="mm-fabric-pill mm-fabric-pill-on">✓</span>
      <span class="mm-fabric-matrix-cell-detail">${escapeHtml(detail)}</span>
    `;
  }
  if (kind === 'stt.transcribe') {
    const detail = [
      cap.in_process ? 'built-in' : 'sidecar',
      cap.streaming ? 'streaming' : 'batch',
      Array.isArray(cap.languages) && cap.languages.length > 0
        ? cap.languages.slice(0, 3).join('/')
        : '',
    ].filter(Boolean).join(' · ');
    return `
      <span class="mm-fabric-pill mm-fabric-pill-on">✓</span>
      <span class="mm-fabric-matrix-cell-detail">${escapeHtml(detail)}</span>
    `;
  }
  if (kind === 'cast.render') {
    const detail = [
      cap.tier || '',
      cap.gpu_model || cap.gpu_vendor || '',
      cap.hw_encoder ? `${cap.hw_encoder}` : '',
    ].filter(Boolean).join(' · ');
    return `
      <span class="mm-fabric-pill mm-fabric-pill-on">✓</span>
      <span class="mm-fabric-matrix-cell-detail">${escapeHtml(detail)}</span>
    `;
  }
  return `<span class="mm-fabric-pill mm-fabric-pill-on">✓</span>`;
}

function _formatCtx(n) {
  if (n >= 1000) return `${Math.round(n / 1024)}k`;
  return String(n);
}

function _formatCount(n) {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

function _renderDiscoverSection() {
  if (!_fabricEnabled) return '';

  const empty = !_lastDiscovery;
  const peers = (_lastDiscovery && _lastDiscovery.peers) || [];
  const selfSeen = (_lastDiscovery && _lastDiscovery.self_seen) || [];
  const alreadyPaired = (_lastDiscovery && _lastDiscovery.already_paired) || [];
  const errors = (_lastDiscovery && _lastDiscovery.errors) || {};
  const hostsProbed = (_lastDiscovery && _lastDiscovery.hosts_probed) || 0;
  const durationS = (_lastDiscovery && _lastDiscovery.duration_s) || 0;
  const subnetInputValue = (_lastDiscovery && _lastDiscovery._lastSubnet) || '';

  const errorEntries = Object.entries(errors);
  const hasResults = !empty && (peers.length || selfSeen.length || alreadyPaired.length || errorEntries.length);

  return `
    <div class="mm-section" id="mm-fabric-discover-section">
      <div class="mm-section-header mm-section-header-stack">
        <div>
          <div class="mm-section-title">Find peers on your LAN</div>
          <div class="mm-section-copy">
            Sweep the local network for other Augmentum instances. Each
            responder's fingerprint shows up as a candidate -- you still
            confirm fingerprints out-of-band and click Pair to finish the
            handshake. Discovery never auto-pairs.
          </div>
        </div>
      </div>
      <div class="mm-fabric-discover-form">
        <label class="mm-fabric-pair-field">
          <span class="mm-fabric-pair-label">Subnet (optional)</span>
          <input class="field-input" type="text" id="mm-fabric-discover-subnet"
                 placeholder="192.168.1.0/24"
                 value="${escapeHtml(subnetInputValue)}">
        </label>
        <button class="btn btn-primary" id="mm-fabric-scan-btn"
                ${_isScanning ? 'disabled' : ''}>
          ${_isScanning ? 'Scanning…' : 'Scan LAN'}
        </button>
        <div class="mm-fabric-hint">
          Leave blank to try the common consumer-router defaults
          (192.168.0.0/24, 192.168.1.0/24, 10.0.0.0/24). Augmentum
          probes only its known ports (6443 HTTPS, 6100 HTTP); non-RFC1918
          ranges and subnets wider than /22 are refused.
        </div>
      </div>
      ${hasResults ? `
        <div class="mm-fabric-discover-summary">
          Probed ${hostsProbed} host${hostsProbed === 1 ? '' : 's'}
          in ${durationS.toFixed(1)}s.
          ${peers.length ? `Found ${peers.length} new candidate${peers.length === 1 ? '' : 's'}.` : 'No new candidates.'}
        </div>
      ` : ''}
      ${peers.length ? `
        <div class="mm-fabric-discover-results">
          ${peers.map((p, idx) => _renderDiscoveredPeerCard(p, idx, 'pairable')).join('')}
        </div>
      ` : ''}
      ${alreadyPaired.length ? `
        <div class="mm-fabric-discover-subhead">Already paired (${alreadyPaired.length})</div>
        <div class="mm-fabric-discover-results">
          ${alreadyPaired.map((p, idx) => _renderDiscoveredPeerCard(p, idx, 'paired')).join('')}
        </div>
      ` : ''}
      ${selfSeen.length ? `
        <div class="mm-fabric-discover-subhead">This node (${selfSeen.length})</div>
        <div class="mm-fabric-discover-results">
          ${selfSeen.map((p, idx) => _renderDiscoveredPeerCard(p, idx, 'self')).join('')}
        </div>
      ` : ''}
      ${errorEntries.length ? `
        <div class="mm-fabric-discover-errors">
          ${errorEntries.map(([k, v]) =>
            `<div class="mm-fabric-discover-error">${escapeHtml(k)}: ${escapeHtml(v)}</div>`,
          ).join('')}
        </div>
      ` : ''}
    </div>
  `;
}

function _renderDiscoveredPeerCard(peer, idx, variant) {
  const icon = peer.icon || DEFAULT_PEER_ICON;
  const labelLine = peer.hostname
    ? `${escapeHtml(peer.hostname)} <span class="mm-fabric-mono">${escapeHtml(peer.addr)}</span>`
    : `<span class="mm-fabric-mono">${escapeHtml(peer.addr)}</span>`;
  let action = '';
  if (variant === 'pairable') {
    action = `
      <button class="btn btn-primary btn-sm mm-fabric-discover-pair"
              data-discover-idx="${idx}">
        Pair
      </button>
    `;
  } else if (variant === 'paired') {
    action = `<span class="mm-fabric-discover-badge mm-fabric-discover-badge-paired">Paired</span>`;
  } else if (variant === 'self') {
    action = `<span class="mm-fabric-discover-badge mm-fabric-discover-badge-self">This node</span>`;
  }
  return `
    <div class="mm-fabric-discover-card">
      <div class="mm-fabric-discover-icon">${escapeHtml(icon)}</div>
      <div class="mm-fabric-discover-meta">
        <div class="mm-fabric-discover-host">${labelLine}</div>
        <div class="mm-fabric-discover-fp mm-fabric-mono">${escapeHtml(peer.fingerprint)}</div>
        ${peer.version ? `<div class="mm-fabric-discover-version">v${escapeHtml(peer.version)}</div>` : ''}
      </div>
      <div class="mm-fabric-discover-action">${action}</div>
    </div>
  `;
}

function _renderPairSection() {
  if (!_fabricEnabled) return '';

  return `
    <div class="mm-section">
      <div class="mm-section-header mm-section-header-stack">
        <div>
          <div class="mm-section-title">Add a peer</div>
          <div class="mm-section-copy">
            Paste the remote node's URL + fingerprint to register them
            as a peer. The remote operator must paste THIS node's
            fingerprint into their own pairing form. Both sides must
            complete the handshake for the link to work.
          </div>
        </div>
      </div>
      <div class="mm-fabric-pair-form">
        <label class="mm-fabric-pair-field">
          <span class="mm-fabric-pair-label">Peer URL</span>
          <input class="field-input" type="text" id="mm-fabric-peer-url"
                 placeholder="https://192.168.1.42:6443 or peer.local:6443">
        </label>
        <label class="mm-fabric-pair-field">
          <span class="mm-fabric-pair-label">Peer fingerprint</span>
          <input class="field-input" type="text" id="mm-fabric-peer-fp"
                 placeholder="SHA256:abcdef...">
        </label>
        <div class="mm-fabric-pair-field">
          <span class="mm-fabric-pair-label">
            Icon <span class="mm-fabric-icon-preview" id="mm-fabric-icon-preview">${escapeHtml(_pickedPeerIcon || DEFAULT_PEER_ICON)}</span>
          </span>
          <div class="mm-fabric-icon-grid" id="mm-fabric-icon-grid">
            ${PEER_ICON_GRID.map((emoji) => `
              <button type="button" class="mm-fabric-icon-btn ${_pickedPeerIcon === emoji ? 'active' : ''}"
                      data-fabric-icon="${escapeHtml(emoji)}"
                      aria-label="Select icon ${escapeHtml(emoji)}">
                ${escapeHtml(emoji)}
              </button>
            `).join('')}
            <input type="text" class="field-input mm-fabric-icon-other"
                   id="mm-fabric-icon-other" maxlength="8"
                   placeholder="Other…"
                   value="${escapeHtml(_pickedPeerIcon && !PEER_ICON_GRID.includes(_pickedPeerIcon) ? _pickedPeerIcon : '')}">
          </div>
          <div class="mm-fabric-hint">
            Your label for this peer in your fleet. Other operators
            pairing the same box can pick a different icon for their
            view — labels are local-only.
          </div>
        </div>
        <button class="btn btn-primary" id="mm-fabric-pair-submit">
          Send pair request
        </button>
        <div class="mm-fabric-hint">
          Pairing flows operator-to-operator: both sides must add each
          other before requests route between them. Verify the fingerprint
          out-of-band (read it aloud / share a screenshot) before pasting --
          the fingerprint is what proves the peer is who they claim. The
          TLS certificate on the remote is intentionally NOT validated
          (self-signed Caddy certs on LAN are normal); peer identity is
          established by the pinned fingerprint + signed envelopes, not
          by the cert chain.
        </div>
      </div>
    </div>
  `;
}

// ── Event wiring ───────────────────────────────────────────────────

function _bindEvents() {
  // Toggle (sync to backend via the standard tool-settings PUT).
  const toggle = document.getElementById('mm-fabric-enabled-toggle');
  if (toggle && !toggle.dataset.bound) {
    toggle.dataset.bound = '1';
    toggle.addEventListener('change', _onToggleChange);
  }

  // Copy fingerprint to clipboard.
  document.querySelectorAll('.mm-fabric-copy[data-copy]').forEach((btn) => {
    if (btn.dataset.bound) return;
    btn.dataset.bound = '1';
    btn.addEventListener('click', () => {
      const text = btn.dataset.copy || '';
      navigator.clipboard.writeText(text).then(
        () => showToast('Fingerprint copied to clipboard', 'success'),
        () => showToast('Could not copy fingerprint', 'error'),
      );
    });
  });

  // Unpair buttons.
  document.querySelectorAll('.mm-fabric-unpair[data-node-id]').forEach((btn) => {
    if (btn.dataset.bound) return;
    btn.dataset.bound = '1';
    btn.addEventListener('click', () => _onUnpairClick(btn.dataset.nodeId));
  });

  // Pair submit.
  const pairBtn = document.getElementById('mm-fabric-pair-submit');
  if (pairBtn && !pairBtn.dataset.bound) {
    pairBtn.dataset.bound = '1';
    pairBtn.addEventListener('click', _onPairSubmit);
  }

  // Scan LAN button.
  const scanBtn = document.getElementById('mm-fabric-scan-btn');
  if (scanBtn && !scanBtn.dataset.bound) {
    scanBtn.dataset.bound = '1';
    scanBtn.addEventListener('click', _onScanLanClick);
  }

  // Per-result Pair buttons — pre-fill the pair form below.
  document.querySelectorAll('.mm-fabric-discover-pair[data-discover-idx]').forEach((btn) => {
    if (btn.dataset.bound) return;
    btn.dataset.bound = '1';
    btn.addEventListener('click', () => {
      const idx = parseInt(btn.dataset.discoverIdx || '-1', 10);
      const peer = (_lastDiscovery && _lastDiscovery.peers && _lastDiscovery.peers[idx]) || null;
      if (peer) _prefillPairForm(peer);
    });
  });

  // Icon picker — curated grid buttons.
  document.querySelectorAll('.mm-fabric-icon-btn[data-fabric-icon]').forEach((btn) => {
    if (btn.dataset.bound) return;
    btn.dataset.bound = '1';
    btn.addEventListener('click', () => {
      _pickedPeerIcon = btn.dataset.fabricIcon || '';
      // Reflect selection: clear "Other" input + repaint active state.
      const other = document.getElementById('mm-fabric-icon-other');
      if (other) other.value = '';
      const preview = document.getElementById('mm-fabric-icon-preview');
      if (preview) preview.textContent = _pickedPeerIcon || DEFAULT_PEER_ICON;
      document.querySelectorAll('.mm-fabric-icon-btn').forEach((b) => {
        b.classList.toggle('active', b.dataset.fabricIcon === _pickedPeerIcon);
      });
    });
  });

  // Icon picker — "Other…" free-form input (pair form, not the
  // local-node identity input).
  const otherInput = document.querySelector('.mm-fabric-pair-form .mm-fabric-icon-other');
  if (otherInput && !otherInput.dataset.bound) {
    otherInput.dataset.bound = '1';
    otherInput.addEventListener('input', () => {
      const val = (otherInput.value || '').trim();
      _pickedPeerIcon = val;
      // Deactivate curated buttons when a custom value is typed.
      document.querySelectorAll('.mm-fabric-icon-btn').forEach((b) => {
        b.classList.toggle('active', !val && b.dataset.fabricIcon === _pickedPeerIcon);
      });
      const preview = document.getElementById('mm-fabric-icon-preview');
      if (preview) preview.textContent = val || DEFAULT_PEER_ICON;
    });
  }

  // Local-node icon input — saves on blur (debounce-free; emoji is
  // small + the operator only picks once usually).
  const localIconInput = document.getElementById('mm-fabric-local-icon-input');
  if (localIconInput && !localIconInput.dataset.bound) {
    localIconInput.dataset.bound = '1';
    localIconInput.addEventListener('input', () => {
      const preview = document.getElementById('mm-fabric-local-icon-preview');
      if (preview) preview.textContent = (localIconInput.value || '').trim() || DEFAULT_PEER_ICON;
    });
    localIconInput.addEventListener('blur', async () => {
      const val = (localIconInput.value || '').trim().slice(0, 8);
      try {
        const resp = await fetch('/api/config/tools', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ local_fabric_icon: val }),
        });
        if (resp.ok) {
          window.__augmentumLocalFabricIcon = val;
          showToast('Local icon saved', 'success');
        }
      } catch (err) {
        showToast('Could not save local icon', 'error');
      }
    });
  }
}

async function _onToggleChange(evt) {
  const enabled = evt.target.checked;
  try {
    const resp = await fetch('/api/config/tools', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fabric_enabled: enabled }),
    });
    if (!resp.ok) throw new Error('save failed');
    showToast(
      enabled
        ? 'Fabric enabled -- restart Augmentum to activate'
        : 'Fabric disabled -- restart Augmentum to deactivate',
      'success',
    );
  } catch (err) {
    // Revert the checkbox if the save failed.
    evt.target.checked = !enabled;
    showToast('Could not save fabric setting', 'error');
  }
}

async function _onUnpairClick(nodeId) {
  if (!nodeId) return;
  const confirmed = window.confirm(
    'Unpair this peer? The remote will go offline until paired again.',
  );
  if (!confirmed) return;
  try {
    const resp = await fetch(`/api/fabric/peers/${encodeURIComponent(nodeId)}`, {
      method: 'DELETE',
    });
    if (!resp.ok) throw new Error(`status ${resp.status}`);
    showToast('Peer unpaired', 'success');
    await refreshFabricState();
  } catch (err) {
    showToast('Could not unpair peer', 'error');
  }
}

async function _onPairSubmit() {
  const urlInput = document.getElementById('mm-fabric-peer-url');
  const fpInput = document.getElementById('mm-fabric-peer-fp');
  const submitBtn = document.getElementById('mm-fabric-pair-submit');
  const url = (urlInput && urlInput.value || '').trim();
  const fingerprint = (fpInput && fpInput.value || '').trim();

  if (!url || !fingerprint) {
    showToast('Both peer URL and fingerprint are required', 'error');
    return;
  }
  if (!fingerprint.startsWith('SHA256:')) {
    showToast('Fingerprint must start with "SHA256:"', 'error');
    return;
  }

  // Disable while in-flight so a double-click can't fire two pair
  // attempts. The backend persist step is idempotent, but two POSTs
  // also means two HTTP round-trips to the remote — wasteful.
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.textContent = 'Pairing…';
  }

  try {
    const resp = await fetch('/api/fabric/pair-with-remote', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        remote_url: url,
        expected_fingerprint: fingerprint,
        icon: _pickedPeerIcon || '',
      }),
    });
    if (resp.ok) {
      const body = await resp.json();
      const remoteId = (body && body.paired && body.paired.node_id) || 'remote peer';
      const iconPrefix = _pickedPeerIcon ? `${_pickedPeerIcon} ` : '';
      showToast(`Paired with ${iconPrefix}${remoteId}`, 'success');
      // Clear the form on success so it's visibly resettable.
      if (urlInput) urlInput.value = '';
      if (fpInput) fpInput.value = '';
      _pickedPeerIcon = '';
      await refreshFabricState();
    } else {
      // The 502 path surfaces operator-facing detail from the remote
      // (e.g. "fingerprint mismatch", "remote unreachable"). Render it
      // verbatim so the operator sees the actual problem.
      let detail = `status ${resp.status}`;
      try {
        const errBody = await resp.json();
        if (errBody && typeof errBody.detail === 'string') {
          detail = errBody.detail;
        }
      } catch (_) { /* response wasn't JSON — keep status-code fallback */ }
      showToast(`Pair failed: ${detail}`, 'error');
    }
  } catch (err) {
    showToast(
      `Pair request errored: ${(err && err.message) || String(err)}`,
      'error',
    );
  } finally {
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Send pair request';
    }
  }
}

async function _onScanLanClick() {
  if (_isScanning) return;
  const subnetInput = document.getElementById('mm-fabric-discover-subnet');
  const subnet = (subnetInput && subnetInput.value || '').trim();

  _isScanning = true;
  // Repaint just to flip the button state to "Scanning…"; cheap.
  await refreshFabricState();

  try {
    const resp = await fetch('/api/fabric/discover', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(subnet ? { subnet } : {}),
    });
    if (!resp.ok) {
      let detail = `status ${resp.status}`;
      try {
        const errBody = await resp.json();
        if (errBody && typeof errBody.detail === 'string') detail = errBody.detail;
      } catch (_) { /* response wasn't JSON — keep status-code fallback */ }
      showToast(`Scan failed: ${detail}`, 'error');
      _lastDiscovery = null;
    } else {
      const body = await resp.json();
      // Tuck the queried subnet onto the discovery result so the input
      // stays populated after the re-render. Underscore-prefixed so it
      // doesn't collide with backend fields.
      body._lastSubnet = subnet;
      _lastDiscovery = body;
      const found = (body.peers && body.peers.length) || 0;
      if (found > 0) {
        showToast(
          `Found ${found} candidate peer${found === 1 ? '' : 's'}`,
          'success',
        );
      } else {
        showToast('Scan complete — no new peers found', 'info');
      }
    }
  } catch (err) {
    showToast(
      `Scan errored: ${(err && err.message) || String(err)}`,
      'error',
    );
    _lastDiscovery = null;
  } finally {
    _isScanning = false;
    await refreshFabricState();
  }
}

function _prefillPairForm(peer) {
  const urlInput = document.getElementById('mm-fabric-peer-url');
  const fpInput = document.getElementById('mm-fabric-peer-fp');
  if (urlInput) urlInput.value = peer.url || '';
  if (fpInput) fpInput.value = peer.fingerprint || '';
  // Adopt the advertised icon as a starting point; operator can override.
  if (peer.icon) {
    _pickedPeerIcon = peer.icon;
    const preview = document.getElementById('mm-fabric-icon-preview');
    if (preview) preview.textContent = peer.icon;
    document.querySelectorAll('.mm-fabric-icon-btn').forEach((b) => {
      b.classList.toggle('active', b.dataset.fabricIcon === peer.icon);
    });
  }
  showToast(
    'Pair form pre-filled — verify the fingerprint matches what the remote operator shared, then click Send pair request.',
    'info',
  );
  // Scroll the pair form into view; on tall screens it would otherwise
  // sit below the fold after a successful scan.
  const pairBtn = document.getElementById('mm-fabric-pair-submit');
  if (pairBtn && pairBtn.scrollIntoView) {
    pairBtn.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}
