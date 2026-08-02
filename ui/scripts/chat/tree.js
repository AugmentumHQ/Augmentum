/* ==========================================================================
   Chat Module — Tree Helpers
   Session tree data structure: node IDs, paths, branching, siblings
   Pure functions — no DOM, no side effects
   ========================================================================== */

export function generateNodeId() {
  return 'n_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

export function getPathToRoot(session, nodeId) {
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

export function getActivePath(session) {
  if (!session || !session.tree) return [];
  // If activeLeafId is missing or points to a deleted node, try to recover
  if (!session.activeLeafId || !session.tree[session.activeLeafId]) {
    const recovered = findAnyLeaf(session);
    if (!recovered) return [];
    session.activeLeafId = recovered;
  }
  return getPathToRoot(session, session.activeLeafId)
    .map(id => session.tree[id])
    .filter(Boolean);
}

/**
 * Find any valid leaf node in the tree (for recovery when activeLeafId is broken).
 * Prefers the deepest leaf from rootId, falls back to any node's deepest leaf.
 */
export function findAnyLeaf(session) {
  if (!session || !session.tree) return null;
  const keys = Object.keys(session.tree);
  if (keys.length === 0) return null;
  // Try from rootId first
  if (session.rootId && session.tree[session.rootId]) {
    return getDeepestLeaf(session, session.rootId);
  }
  // Fallback: find any root-like node (no parent or parent missing)
  for (const id of keys) {
    const node = session.tree[id];
    if (!node.parentId || !session.tree[node.parentId]) {
      return getDeepestLeaf(session, id);
    }
  }
  // Last resort: just use the first node
  return getDeepestLeaf(session, keys[0]);
}

// Strip any rendered scene-image markup from prior assistant turns before
// sending history to the LLM. The model has been observed imitating the
// rendered image card (sometimes with malformed attributes that leak as
// visible text) when it sees one in context. Replacing the URL with a
// short opaque placeholder removes the imitation surface entirely while
// preserving the *fact* that an image was emitted at this point in the
// turn — useful for narrative continuity.
const _SCENE_IMG_MD_RE = /!\[([^\]]*)\]\((\/api\/(?:image|artifacts)\/[a-zA-Z0-9_-]+(?:\/[a-zA-Z]+)?)\)/g;
const _SCENE_IMG_TAG_RE = /<img\s+[^>]*src=["']\/api\/(?:image|artifacts)\/[a-zA-Z0-9_-]+(?:\/[a-zA-Z]+)?["'][^>]*>/gi;

function _redactSceneImages(content) {
  if (typeof content !== 'string' || !content) return content;
  let out = content.replace(_SCENE_IMG_MD_RE, (_m, alt) => {
    const label = (alt || '').trim() || 'scene';
    return `[image: ${label}]`;
  });
  out = out.replace(_SCENE_IMG_TAG_RE, '[image: scene]');
  return out;
}

export function buildMessagesForAPI(session) {
  return getActivePath(session)
    .filter(node => node.role !== 'image')
    .map(node => {
      // Redact only assistant turns — user-supplied images stay as-is so
      // multimodal attachments still reach VL backends below.
      const rawContent = node.role === 'assistant'
        ? _redactSceneImages(node.content)
        : node.content;
      const msg = { role: node.role, content: rawContent };
      if (node.images && node.images.length > 0) {
        const realImages = node.images.filter(i => i && i !== '[image]');
        if (realImages.length > 0) {
          const parts = [];
          if (rawContent) {
            parts.push({ type: 'text', text: rawContent });
          }
          for (const img of realImages) {
            parts.push({ type: 'image_url', image_url: { url: img } });
          }
          msg.content = parts;
        }
      }
      // DeepSeek and other reasoning models 400 if a previous assistant
      // turn had reasoning_content and the next request omits it on
      // replay. node.reasoning.thinking holds the model-native reasoning
      // tokens captured during streaming; emit them back so the
      // round-trip closes. Other providers ignore the unknown field.
      if (node.role === 'assistant' && node.reasoning && node.reasoning.thinking) {
        msg.reasoning_content = node.reasoning.thinking;
      }
      return msg;
    });
}

export function addChildNode(session, parentId, role, content) {
  if (!session.tree) session.tree = {};
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
  if (!session.rootId) session.rootId = id;
  return node;
}

export function sessionHasMessages(session) {
  return session && session.tree && Object.keys(session.tree).length > 0;
}

export function getSiblingInfo(session, nodeId) {
  const node = session.tree[nodeId];
  if (!node || !node.parentId) return null;
  const parent = session.tree[node.parentId];
  if (!parent) return null;
  const siblings = parent.children.filter(cid => {
    const child = session.tree[cid];
    return child && child.role === node.role;
  });
  const idx = siblings.indexOf(nodeId);
  if (siblings.length <= 1) return null;
  return { siblings, index: idx, total: siblings.length };
}

export function switchToSibling(session, nodeId, direction) {
  const info = getSiblingInfo(session, nodeId);
  if (!info) return;
  const newIdx = info.index + direction;
  if (newIdx < 0 || newIdx >= info.total) return;
  const newId = info.siblings[newIdx];
  session.activeLeafId = getDeepestLeaf(session, newId);
}

export function getDeepestLeaf(session, nodeId) {
  const node = session.tree[nodeId];
  if (!node || !node.children || node.children.length === 0) return nodeId;
  return getDeepestLeaf(session, node.children[node.children.length - 1]);
}

export function countDescendants(session, nodeId) {
  const node = session.tree[nodeId];
  if (!node || !node.children) return 0;
  let count = node.children.length;
  for (const cid of node.children) {
    count += countDescendants(session, cid);
  }
  return count;
}

export function removeNodeAndDescendants(session, nodeId) {
  const node = session.tree[nodeId];
  if (!node) return;
  if (node.parentId && session.tree[node.parentId]) {
    const parent = session.tree[node.parentId];
    parent.children = parent.children.filter(id => id !== nodeId);
  }
  const toRemove = [nodeId];
  while (toRemove.length > 0) {
    const id = toRemove.pop();
    const n = session.tree[id];
    if (n && n.children) {
      toRemove.push(...n.children);
    }
    delete session.tree[id];
  }
  if (session.rootId === nodeId) {
    session.rootId = null;
  }
}

export function migrateSessionToV2(session) {
  const tree = {};
  let prevId = null;
  const messages = session.messages || [];
  messages.forEach((msg, i) => {
    const id = 'n_migrated_' + i;
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
    prevId = id;
  });
  return {
    id: session.id,
    title: session.title || 'Migrated Chat',
    version: 2,
    tree,
    rootId: messages.length > 0 ? 'n_migrated_0' : null,
    activeLeafId: prevId,
    createdAt: session.createdAt || Date.now(),
  };
}

/**
 * Scan session lorebook entries against recent messages.
 * Returns { before: string[], after: string[] } content arrays
 * for injection before/after the character prompt.
 */
export function scanLorebook(session, chatMessages) {
  const before = [];
  const after = [];
  const state = session.lorebookState || {};
  const entries = session.lorebook || [];

  // Combine recent message content for keyword scanning
  const recentText = chatMessages.slice(-6).map(m => m.content).join(' ').toLowerCase();

  // Sort by priority (lower number = higher priority)
  const sorted = entries
    .map((e, i) => ({ ...e, _idx: i }))
    .sort((a, b) => (a.priority || 100) - (b.priority || 100));

  for (const entry of sorted) {
    if (!entry.content) continue;

    const idx = entry._idx;
    const es = state[idx] || { stickyRemaining: 0, cooldownRemaining: 0 };

    // Constant entries always inject
    if (entry.constant) {
      (entry.position === 'before_char' ? before : after).push(entry.content);
      continue;
    }

    // If on cooldown, decrement and skip
    if (es.cooldownRemaining > 0) {
      es.cooldownRemaining--;
      state[idx] = es;
      continue;
    }

    // Check if still sticky (active from prior trigger)
    if (es.stickyRemaining > 0) {
      (entry.position === 'before_char' ? before : after).push(entry.content);
      es.stickyRemaining--;
      // Start cooldown when sticky expires
      if (es.stickyRemaining <= 0 && entry.cooldown_turns > 0) {
        es.cooldownRemaining = entry.cooldown_turns;
      }
      state[idx] = es;
      continue;
    }

    // Keyword scan
    const keys = entry.keys || [];
    const triggered = keys.length > 0 && keys.some(k => k && recentText.includes(k.toLowerCase()));

    if (triggered) {
      (entry.position === 'before_char' ? before : after).push(entry.content);
      // Activate sticky if configured
      if (entry.sticky_turns > 0) {
        es.stickyRemaining = entry.sticky_turns;
      } else if (entry.cooldown_turns > 0) {
        // No sticky, start cooldown immediately after one-shot
        es.cooldownRemaining = entry.cooldown_turns;
      }
      state[idx] = es;
    }
  }

  session.lorebookState = state;
  return { before, after };
}
