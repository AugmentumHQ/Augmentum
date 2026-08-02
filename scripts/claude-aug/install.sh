#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# install.sh — set up `claude-aug` (Claude Code → local Augmentum)
#
#   ./scripts/claude-aug/install.sh
#   ./scripts/claude-aug/install.sh --api-key sk-aug-... --base-url http://localhost:6100
#
# Idempotent: re-run any time. Existing claude.env values are kept as
# defaults so re-installing never wipes your config.
# ────────────────────────────────────────────────────────────────────
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUG_DIR="$HOME/.augmentum"
CFG_DIR="$AUG_DIR/claude-config"
ENV_FILE="$AUG_DIR/claude.env"

env_val() {  # env_val NAME DEFAULT
    if [[ -f "$ENV_FILE" ]]; then
        local v
        v=$(grep -E "^export $1=" "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"'"'" || true)
        [[ -n "$v" ]] && { echo "$v"; return; }
    fi
    echo "$2"
}

API_KEY="" BASE_URL="" MAIN_MODEL="" SMALL_MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-key)     API_KEY="$2"; shift 2 ;;
        --base-url)    BASE_URL="$2"; shift 2 ;;
        --main-model)  MAIN_MODEL="$2"; shift 2 ;;
        --small-model) SMALL_MODEL="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done
API_KEY="${API_KEY:-$(env_val AUGMENTUM_API_KEY '')}"
BASE_URL="${BASE_URL:-$(env_val ANTHROPIC_BASE_URL http://localhost:6100)}"
MAIN_MODEL="${MAIN_MODEL:-$(env_val ANTHROPIC_MODEL deepseek-v4-pro)}"
SMALL_MODEL="${SMALL_MODEL:-$(env_val ANTHROPIC_SMALL_FAST_MODEL deepseek-v4-flash)}"

if [[ -z "$API_KEY" ]]; then
    read -rp "Augmentum API key (Settings → API Keys): " API_KEY
    [[ -z "$API_KEY" ]] && { echo "API key is required"; exit 1; }
fi

mkdir -p "$AUG_DIR" "$CFG_DIR"

# ── 1. claude.env from template ─────────────────────────────────────
sed -e "s|{{API_KEY}}|$API_KEY|g" \
    -e "s|{{BASE_URL}}|$BASE_URL|g" \
    -e "s|{{MAIN_MODEL}}|$MAIN_MODEL|g" \
    -e "s|{{SMALL_MODEL}}|$SMALL_MODEL|g" \
    "$SRC/claude.env.template" > "$ENV_FILE"

# ── 2. Copy portable files ──────────────────────────────────────────
cp "$SRC/claude-aug.sh" "$SRC/claude-aug.ps1" "$SRC/atp-mcp-bridge.py" \
   "$SRC/bridge-hooks.py" "$SRC/doctor.py" "$AUG_DIR/"
cp "$SRC/statusline.ps1" "$SRC/CLAUDE.md" "$CFG_DIR/"
cp -r "$SRC/skills" "$CFG_DIR/"
chmod +x "$AUG_DIR/claude-aug.sh" "$AUG_DIR/atp-mcp-bridge.py" "$AUG_DIR/bridge-hooks.py"

# ── 3. Generate claude-config (paths + key hash are user-specific) ──
BRIDGE="$AUG_DIR/atp-mcp-bridge.py"
HOOKS="$AUG_DIR/bridge-hooks.py"
KEY_HASH="${API_KEY: -20}"   # CC identifies approved keys by last 20 chars

cat > "$CFG_DIR/.claude.json" <<EOF
{
  "hasCompletedOnboarding": true,
  "customApiKeyResponses": { "approved": ["$KEY_HASH"], "rejected": [] },
  "mcpServers": {
    "atp": { "type": "stdio", "command": "python3", "args": ["$BRIDGE"] }
  }
}
EOF

cat > "$CFG_DIR/settings.json" <<EOF
{
  "statusLine": {
    "type": "command",
    "command": "printf 'AUG | %s @ %s' \"\${ANTHROPIC_MODEL:-?}\" \"\${ANTHROPIC_BASE_URL#*://}\""
  },
  "permissions": {
    "deny": ["WebSearch", "WebFetch"],
    "allow": ["mcp__atp"]
  },
  "hooks": {
    "SessionStart": [
      { "matcher": "", "hooks": [ { "type": "command", "command": "python3 $HOOKS --hook SessionStart" } ] }
    ],
    "Stop": [
      { "matcher": "", "hooks": [ { "type": "command", "command": "python3 $HOOKS --hook Stop" } ] }
    ],
    "PreToolUse": [
      { "matcher": "Bash|Edit|Write|NotebookEdit|Agent|Workflow|Task", "hooks": [ { "type": "command", "command": "python3 $HOOKS --hook PreToolUse" } ] }
    ],
    "Notification": [
      { "matcher": "permission", "hooks": [ { "type": "command", "command": "python3 $HOOKS --hook PreToolUse" } ] }
    ]
  }
}
EOF

# ── 4. Detect the Augmentum container (for wrapper auto-start) ──────
CONTAINER="$(docker ps -a --format '{{.Names}}' 2>/dev/null | grep -E '^augmentum.*augmentum' | head -1 || true)"
if [[ -n "$CONTAINER" && "$CONTAINER" != "augmentum-augmentum-1" ]]; then
    sed -i.bak "s/augmentum-augmentum-1/$CONTAINER/g" "$AUG_DIR/claude-aug.ps1" && rm -f "$AUG_DIR/claude-aug.ps1.bak"
    echo "Wrapper auto-start wired to container: $CONTAINER"
fi

# ── 5. Hook into shell rc ───────────────────────────────────────────
RC="$HOME/.bashrc"; [[ "${SHELL:-}" == */zsh ]] && RC="$HOME/.zshrc"
if ! grep -q 'claude-aug' "$RC" 2>/dev/null; then
    printf '\n# Claude Code via local Augmentum\nalias claude-aug="%s"\n' "$AUG_DIR/claude-aug.sh" >> "$RC"
    echo "Added claude-aug alias to $RC"
fi

# ── 6. Remind about server-side aliases ─────────────────────────────
cat <<EOF

Installed. Open a NEW terminal and run: claude-aug

One server-side step (needed once): ensure these are in Augmentum's .env
so the container maps Claude Code's hardcoded claude-* model IDs:
  AUGMENTUM_ANTHROPIC_ALIAS_HAIKU=$SMALL_MODEL
  AUGMENTUM_ANTHROPIC_ALIAS_SONNET=$SMALL_MODEL
  AUGMENTUM_ANTHROPIC_ALIAS_OPUS=$MAIN_MODEL
  AUGMENTUM_ANTHROPIC_ALIAS_DEFAULT=$MAIN_MODEL
EOF
