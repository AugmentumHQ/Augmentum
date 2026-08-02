#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# claude-aug — Claude Code routed through your LOCAL Augmentum proxy.
#
#   claude-aug                              # default models from claude.env
#   claude-aug models                       # list available Augmentum models
#   claude-aug --model qwen3.6-35b          # one-off main model switch
#   claude-aug --small deepseek-v4-flash    # one-off subagent model
#   claude-aug -p "summarize this repo"     # any claude args pass through
#
# Normal `claude` is untouched — still hits Anthropic with your sub.
# Configuration: ~/.augmentum/claude.env
# ────────────────────────────────────────────────────────────────────
set -euo pipefail

# shellcheck disable=SC1090
source "$HOME/.augmentum/claude.env"

# ── Defaults ───────────────────────────────────────────────────────
BASE_URL="${ANTHROPIC_BASE_URL:-http://localhost:6100}"
API_KEY="${ANTHROPIC_API_KEY:-${AUGMENTUM_API_KEY:-}}"
MODEL="${ANTHROPIC_MODEL:-deepseek-v4-flash}"
SMALL_MODEL="${ANTHROPIC_SMALL_FAST_MODEL:-deepseek-v4-flash}"

# ── Parse claude-aug-specific flags ─────────────────────────────────
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        models)
            # List available models from Augmentum
            echo "Fetching models from ${BASE_URL} ..."
            curl -sf "${BASE_URL}/v1/models" \
                -H "x-api-key: ${API_KEY}" \
                -H "anthropic-version: 2023-06-01" \
            | python3 -c "
import json, sys
d = json.load(sys.stdin)
models = [m['id'] for m in d.get('data', []) if not m['id'].startswith(('a/','n/'))]
for m in sorted(set(models)):
    print(f'  {m}')
print(f'\n{len(set(models))} models available')
" 2>/dev/null || echo "❌ Could not reach Augmentum at ${BASE_URL}"
            exit 0
            ;;
        doctor)
            shift
            exec python3 "$HOME/.augmentum/doctor.py" "$@"
            ;;
        --profile)
            shift
            if [[ $# -eq 0 ]]; then echo "Missing profile name after --profile"; exit 1; fi
            PROFILE_VAR="AUGMENTUM_PROFILE_$(echo "$1" | tr '[:lower:]' '[:upper:]')"
            PROFILE_VAL="${!PROFILE_VAR:-}"
            if [[ -z "$PROFILE_VAL" ]]; then
                echo "Unknown profile '$1' — define $PROFILE_VAR in claude.env"; exit 1
            fi
            read -r MODEL SMALL_MODEL <<< "$PROFILE_VAL"
            SMALL_MODEL="${SMALL_MODEL:-$MODEL}"
            shift
            ;;
        --model)
            shift
            if [[ $# -eq 0 ]]; then echo "Missing model name after --model"; exit 1; fi
            MODEL="$1"
            shift
            ;;
        --small)
            shift
            if [[ $# -eq 0 ]]; then echo "Missing model name after --small"; exit 1; fi
            SMALL_MODEL="$1"
            shift
            ;;
        --help|-h)
            echo "Usage: claude-aug [--profile NAME] [--model MODEL] [--small MODEL] [models|--help] [claude args...]"
            echo ""
            echo "  models          List available Augmentum models"
            echo "  --profile NAME  Use a model profile (deep/fast/mixed, from claude.env)"
            echo "  --model NAME    Override main model for this session"
            echo "  --small NAME    Override subagent model for this session"
            echo "  --help          This message"
            echo ""
            echo "Default config:  ~/.augmentum/claude.env"
            echo "Normal \`claude\` is untouched — still hits Anthropic directly."
            exit 0
            ;;
        *)
            PASSTHROUGH+=("$1")
            shift
            ;;
    esac
done

# ── Health check ───────────────────────────────────────────────────
if ! curl -sf -o /dev/null "${BASE_URL}/v1/models" \
    -H "x-api-key: ${API_KEY}" \
    -H "anthropic-version: 2023-06-01" 2>/dev/null; then
    echo "ERROR: Augmentum is not reachable at ${BASE_URL}"
    echo "   Start it with: docker compose start"
    exit 1
fi

# ── Status banner ──────────────────────────────────────────────────
echo "+-- Augmentum ----------------------------------------------------+"
printf "|  %-12s %-50s |\n" "Server:" "${BASE_URL}"
printf "|  %-12s %-50s |\n" "Main:" "${MODEL}"
printf "|  %-12s %-50s |\n" "Subagents:" "${SMALL_MODEL}"
printf "|  %-12s %-50s |\n" "Aliases:" "haiku->${AUGMENTUM_ANTHROPIC_ALIAS_HAIKU:-$SMALL_MODEL}  sonnet->${AUGMENTUM_ANTHROPIC_ALIAS_SONNET:-$MODEL}  opus->${AUGMENTUM_ANTHROPIC_ALIAS_OPUS:-$MODEL}"
echo "+-----------------------------------------------------------------+"

# ── Route to Augmentum ─────────────────────────────────────────────
export ANTHROPIC_BASE_URL="${BASE_URL}"
export ANTHROPIC_API_KEY="${API_KEY}"
# (ANTHROPIC_AUTH_TOKEN no longer set: the pre-approved key in claude-config
# handles the OAuth prompt; setting both triggers a dual-auth warning.)
export ANTHROPIC_MODEL="${MODEL}"
export ANTHROPIC_SMALL_FAST_MODEL="${SMALL_MODEL}"
export CLAUDE_CODE_SUBAGENT_MODEL="${SMALL_MODEL}"  # actual subagent model control
# Harness + project identity: project slug scopes server-side memory
# (proxy/harness.py) so conventions/facts don't bleed across projects.
export ANTHROPIC_CUSTOM_HEADERS="X-Augmentum-Harness: claude_code
X-Augmentum-Project: $(basename "$PWD")"

# Fully isolated config: own ~/.claude.json, settings, sessions
export CLAUDE_CONFIG_DIR="$HOME/.augmentum/claude-config"
# No stray Anthropic calls (telemetry, update checks) while on local backend
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
# Local models are slower than the API — allow long generations
export API_TIMEOUT_MS=600000
# ATP research/browser calls can run 1-2 min; don't let CC's MCP timeout
# kill them mid-flight
export MCP_TOOL_TIMEOUT=300000

exec claude "${PASSTHROUGH[@]}"
