#!/usr/bin/env bash
# ============================================================================
# Augmentum Setup — Interactive configuration wizard
# Run once to configure, then use ./start.sh to launch.
# Re-run anytime to reconfigure (your .env customizations are preserved).
# ============================================================================
set -euo pipefail

# --- Terminal capabilities -------------------------------------------------

# Colors (fallback to plain if terminal doesn't support)
if [ -t 1 ] && command -v tput &>/dev/null && [ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]; then
  BOLD=$(tput bold)
  DIM=$(tput dim)
  CYAN=$(tput setaf 6)
  GREEN=$(tput setaf 2)
  YELLOW=$(tput setaf 3)
  RED=$(tput setaf 1)
  RESET=$(tput sgr0)
else
  BOLD="" DIM="" CYAN="" GREEN="" YELLOW="" RED="" RESET=""
fi

# Glyphs (ASCII fallback when the locale isn't UTF-8)
case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in
  *[Uu][Tt][Ff]*8*|*[Uu][Tt][Ff]-8*)
    CHECK="✓" ARROW="❯" RULE="─" CHEV_L="╱" CHEV_R="╲" ;;
  *)
    CHECK="*" ARROW=">" RULE="-" CHEV_L="/" CHEV_R="\\" ;;
esac

hr() { printf '  %s%s%s\n' "$DIM" "$(printf "%${1:-46}s" '' | tr ' ' "$RULE")" "$RESET"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONF_FILE="$SCRIPT_DIR/.augmentum.conf"
ENV_FILE="$SCRIPT_DIR/.env"

TOTAL_STEPS=8

# --- Output helpers --------------------------------------------------------

print_header() {
  echo ""
  echo "      ${CYAN}${CHEV_L}${CHEV_R}${RESET}"
  echo "     ${CYAN}${CHEV_L}${CHEV_R}${CHEV_L}${CHEV_R}${RESET}     ${BOLD}Augmentum${RESET}"
  echo "    ${CYAN}${CHEV_L}${CHEV_R}${CHEV_L}${CHEV_R}${CHEV_L}${CHEV_R}${RESET}    ${DIM}Personal AI on hardware you own${RESET}"
  echo ""
  hr
  echo "  ${DIM}${TOTAL_STEPS} steps ${RULE} press Enter to accept the recommended default${RESET}"
  hr
}

print_step() {
  echo ""
  echo ""
  echo "  ${DIM}$1/$2${RESET}  ${BOLD}$3${RESET}"
}

ok()   { echo "  ${GREEN}${CHECK}${RESET} $1"; }
note() { echo "    ${DIM}$1${RESET}"; }
warn() { echo "    ${YELLOW}$1${RESET}"; }

# Selections recorded for the final summary panel.
SUMMARY_KEYS=()
SUMMARY_VALS=()
add_summary() { SUMMARY_KEYS+=("$1"); SUMMARY_VALS+=("$2"); }

# prompt_choice "<prompt>" <default 1-based> "opt1" "opt2" ...
# Enter accepts the default. Result in PROMPT_CHOICE (0-based).
prompt_choice() {
  local prompt="$1" default="$2"
  shift 2
  local options=("$@")
  local count=${#options[@]}

  echo ""
  for i in "${!options[@]}"; do
    if [ "$((i + 1))" -eq "$default" ]; then
      echo "    ${BOLD}${CYAN}$((i + 1))${RESET}  ${options[$i]}"
    else
      echo "    ${BOLD}$((i + 1))${RESET}  ${options[$i]}"
    fi
  done
  echo ""

  while true; do
    printf "  ${CYAN}${ARROW}${RESET} ${prompt} ${DIM}[1-${count} ${RULE} Enter = ${default}]${RESET} "
    read -r choice
    choice="${choice:-$default}"
    if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$count" ]; then
      # Carry the choice in a global, NOT the exit code: under `set -e` a
      # non-zero `return` kills the script before the caller reads $?
      # (so any non-default selection would abort setup). Always return 0.
      PROMPT_CHOICE=$((choice - 1))
      return 0
    fi
    echo "    ${RED}Please enter a number between 1 and ${count}${RESET}"
  done
}

prompt_yn() {
  local prompt="$1"
  local default="${2:-y}"
  local hint="[Y/n]"
  [ "$default" = "n" ] && hint="[y/N]"

  printf "  ${CYAN}${ARROW}${RESET} ${prompt} ${DIM}${hint}${RESET} "
  read -r answer
  answer="${answer:-$default}"
  [[ "$answer" =~ ^[Yy] ]]
}

prompt_input() {
  local prompt="$1"
  local default="${2:-}"
  if [ -n "$default" ]; then
    printf "  ${CYAN}${ARROW}${RESET} ${prompt} ${DIM}[${default}]${RESET} " >&2
  else
    printf "  ${CYAN}${ARROW}${RESET} ${prompt} " >&2
  fi
  read -r input
  echo "${input:-$default}"
}

# --- Main ------------------------------------------------------------------

print_header

# Show existing config if reconfiguring
if [ -f "$CONF_FILE" ]; then
  EXISTING=$(cat "$CONF_FILE")
  echo ""
  echo "  ${YELLOW}Existing configuration found:${RESET}"
  echo "    ${DIM}${EXISTING}${RESET}"
  echo ""
  if ! prompt_yn "Reconfigure?" "y"; then
    echo ""
    note "Keeping existing configuration. Run ./start.sh to launch."
    echo ""
    exit 0
  fi
fi

# Hardware sniff (used for the GPU/CPU default; never decides silently)
GPU_NAME=""
GPU_VRAM_MB=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | sed 's/^ *//' || true)
  # Total VRAM (MB) — used to default the classifier model tier. Empty/odd
  # output on some drivers → treat as 0 (conservative default, never crash).
  GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -dc '0-9' || true)
  [ -z "$GPU_VRAM_MB" ] && GPU_VRAM_MB=0
fi

# ============================================================
# Step 1: Install type
# ============================================================
print_step 1 $TOTAL_STEPS "Install type"
note "Use Augmentum, or work on it?"

prompt_choice "Install type" 1 \
  "Standard — prebuilt image, no compile step (recommended)" \
  "Contributor — build from source, live-edit ./augmentum and ./ui"

if [ "$PROMPT_CHOICE" -eq 0 ]; then
  # Pull-only: compose.yaml's image: directive fetches from GHCR. The
  # bundled llama-server engine ships inside both image variants, so
  # nothing is lost by not building.
  COMPOSE_FILES="compose.yaml"
  ok "Standard install"
  add_summary "Install" "Standard (prebuilt image)"
else
  # Dev overlay re-adds build: + source bind mounts (+ SYS_PTRACE for
  # py-spy). First start compiles the image locally — expect minutes,
  # not seconds.
  COMPOSE_FILES="compose.yaml compose.dev.yaml"
  ok "Contributor install — first start builds the image (several minutes)"
  add_summary "Install" "Contributor (build from source)"
fi

# ============================================================
# Step 2: Backend selection
# ============================================================
print_step 2 $TOTAL_STEPS "Model backend"
note "How will you run LLMs?"

prompt_choice "Backend" 1 \
  "Built-in engine — bundled llama-server, models managed in the UI (recommended)" \
  "External — I already run Ollama, LM Studio, or another LLM server"

BACKEND_CHOICE=$PROMPT_CHOICE

ENV_VARS=()

case $BACKEND_CHOICE in
  0)
    # Built-in engine (subprocess inside augmentum container)
    ok "Built-in engine"

    # Hardware tier — picks which variant image gets pulled and which
    # Dockerfile is built locally (compose.dev.yaml reads these envs).
    # Default follows nvidia-smi when present; the user always confirms.
    HW_DEFAULT=2
    if [ -n "$GPU_NAME" ]; then
      HW_DEFAULT=1
      note "Detected: $GPU_NAME"
    fi

    prompt_choice "Hardware" $HW_DEFAULT \
      "GPU (NVIDIA) — local image generation + faster LLM inference" \
      "CPU only — works without an NVIDIA GPU"

    if [ "$PROMPT_CHOICE" -eq 0 ]; then
      ENV_VARS+=("AUGMENTUM_VARIANT=gpu")
      ENV_VARS+=("AUGMENTUM_DOCKERFILE=Dockerfile.gpu")
      COMPOSE_FILES="$COMPOSE_FILES compose.gpu.yaml"
      ok "GPU variant"
      add_summary "Backend" "Built-in engine ${RULE} GPU${GPU_NAME:+ ($GPU_NAME)}"
    else
      ENV_VARS+=("AUGMENTUM_VARIANT=cpu")
      ENV_VARS+=("AUGMENTUM_DOCKERFILE=Dockerfile")
      ok "CPU variant"
      note "Image generation needs a GPU — cloud providers work via Settings."
      add_summary "Backend" "Built-in engine ${RULE} CPU"
    fi

    ENV_VARS+=("AUGMENTUM_DEFAULT_BACKEND=engine")
    ENV_VARS+=("AUGMENTUM_ENGINE_MANAGED=true")

    echo ""
    note "Augmentum auto-discovers GGUF models (LM Studio, HuggingFace cache)."
    note "Optionally mount another folder of .gguf files now:"
    MODEL_DIR=$(prompt_input "Path to GGUF models (Enter to skip)" "")
    if [ -n "$MODEL_DIR" ]; then
      ENV_VARS+=("AUGMENTUM_ENGINE_MODEL_DIR=/data/host-models")
      ENGINE_MODEL_MOUNT="$MODEL_DIR"
      ok "Will mount $MODEL_DIR into the container"
      add_summary "Model dir" "$MODEL_DIR"
    else
      # Skip means "no change", not "remove": a pre-existing model-dir
      # config (possibly pointing at a hand-edited mount override) must
      # survive the re-run.
      _old_model_dir=""
      if [ -f "$ENV_FILE" ]; then
        _old_model_dir=$(grep -E '^AUGMENTUM_ENGINE_MODEL_DIR=' "$ENV_FILE" 2>/dev/null | head -n1 | cut -d= -f2- || true)
      fi
      if [ -n "$_old_model_dir" ]; then
        ENV_VARS+=("AUGMENTUM_ENGINE_MODEL_DIR=$_old_model_dir")
        ok "Keeping existing model directory config ($_old_model_dir)"
        add_summary "Model dir" "$_old_model_dir (kept)"
      else
        note "Skipped. Add directories later in Settings > Manage Providers."
      fi
    fi
    ;;
  1)
    # External — user runs their own inference server
    ok "External backend — local servers are auto-detected on startup"

    prompt_choice "Default provider" 1 \
      "Skip — auto-detect, or configure in the UI later" \
      "OpenAI-compatible API (LM Studio, OpenRouter, Together, ...)" \
      "OpenAI (official)"

    PROVIDER_CHOICE=$PROMPT_CHOICE

    case $PROVIDER_CHOICE in
      0)
        add_summary "Backend" "External (auto-detect)"
        ;;
      1)
        API_URL=$(prompt_input "API base URL (e.g. http://localhost:1234/v1)")
        if [ -n "$API_URL" ]; then
          API_KEY=$(prompt_input "API key (Enter to skip)" "")
          ENV_VARS+=("AUGMENTUM_OPENAI_BASE_URL=$API_URL")
          [ -n "$API_KEY" ] && ENV_VARS+=("AUGMENTUM_OPENAI_API_KEY=$API_KEY")
          ENV_VARS+=("AUGMENTUM_DEFAULT_BACKEND=openai")
          ok "Provider configured: $API_URL"
          add_summary "Backend" "External ${RULE} $API_URL"
        else
          add_summary "Backend" "External (auto-detect)"
        fi
        ;;
      2)
        API_KEY=$(prompt_input "OpenAI API key")
        if [ -n "$API_KEY" ]; then
          ENV_VARS+=("AUGMENTUM_OPENAI_API_KEY=$API_KEY")
          ENV_VARS+=("AUGMENTUM_DEFAULT_BACKEND=openai")
          ok "OpenAI configured"
          add_summary "Backend" "External ${RULE} OpenAI"
        else
          add_summary "Backend" "External (auto-detect)"
        fi
        ;;
    esac
    ;;
esac

# ============================================================
# Step 3: Image generation
# ============================================================
print_step 3 $TOTAL_STEPS "Image generation"
note "Local Stable Diffusion / FLUX. Needs an NVIDIA GPU with 4+ GB VRAM."

if prompt_yn "Enable image generation?" "n"; then
  COMPOSE_FILES="$COMPOSE_FILES compose.gpu.yaml"
  ok "Image generation enabled"
  add_summary "Image gen" "on (local SD/FLUX)"

  HF_TOKEN=$(prompt_input "HuggingFace token (optional, for gated models — Enter to skip)" "")
  [ -n "$HF_TOKEN" ] && ENV_VARS+=("AUGMENTUM_HUGGINGFACE_TOKEN=$HF_TOKEN")
else
  ok "Skipped — re-run setup anytime to enable"
  add_summary "Image gen" "off"
fi

# ============================================================
# Step 4: Speech-to-Text
# ============================================================
print_step 4 $TOTAL_STEPS "Speech-to-text"
note "Moonshine (English, streaming) is built in — nothing to add for English."

prompt_choice "Voice input" 1 \
  "English only — built-in Moonshine, no extra container (recommended)" \
  "Multilingual — faster-whisper via Speaches (99 languages, ~2 GB)"

STT_CHOICE=$PROMPT_CHOICE
STT_WEBUI=""

case $STT_CHOICE in
  0)
    ok "Built-in Moonshine (English, streaming)"
    add_summary "Speech-to-text" "Moonshine (built-in)"
    ;;
  1)
    COMPOSE_FILES="$COMPOSE_FILES compose.speaches.yaml"
    ENV_VARS+=("AUGMENTUM_STT_PROVIDER_URL=http://speaches:8000")
    ENV_VARS+=("AUGMENTUM_STT_DEFAULT_MODEL=Systran/faster-whisper-small")
    ENV_VARS+=("AUGMENTUM_STT_MODEL=Systran/faster-whisper-small")
    ENV_VARS+=("AUGMENTUM_VOICE_MOONSHINE_ENABLED=false")
    STT_WEBUI="http://localhost:6200"
    ok "Multilingual STT added (faster-whisper-small, 99 languages)"
    add_summary "Speech-to-text" "faster-whisper (99 languages)"
    ;;
esac

# ============================================================
# Step 5: Text-to-Speech
# ============================================================
print_step 5 $TOTAL_STEPS "Text-to-speech"
note "Kokoro (CPU, 54 voices, mixing) is built in — extras add cloning/emotion."

prompt_choice "Voice output" 1 \
  "Built-in only — Kokoro, no extra container (recommended)" \
  "Chatterbox — multilingual cloning, GPU recommended (~4 GB VRAM)" \
  "Chatterbox Turbo — fast English cloning, GPU recommended (~2 GB VRAM)" \
  "Fish Speech — emotion + cloning, GPU required (~8 GB VRAM, gated)" \
  "Qwen3 TTS — emotion-aware speech, GPU required (~6 GB VRAM)" \
  "Sesame CSM — companion's relational voice, GPU (~3-4 GB VRAM, gated)"

TTS_CHOICE=$PROMPT_CHOICE
TTS_WEBUI=""

case $TTS_CHOICE in
  0)
    ok "Built-in Kokoro (CPU, voice mixing)"
    add_summary "Text-to-speech" "Kokoro (built-in)"
    ;;
  1)
    COMPOSE_FILES="$COMPOSE_FILES compose.chatterbox.yaml"
    ENV_VARS+=("AUGMENTUM_TTS_CHATTERBOX_URL=http://chatterbox:4123")
    ENV_VARS+=("AUGMENTUM_CHATTERBOX_DEVICE=auto")
    TTS_WEBUI="http://localhost:6400"
    ok "Chatterbox added — Kokoro stays as fallback"
    note "500M multilingual model: 23+ languages, highest-quality cloning."
    note "Upload voice samples in Settings > Voice or the Chatterbox WebUI."
    add_summary "Text-to-speech" "Chatterbox + Kokoro fallback"
    ;;
  2)
    COMPOSE_FILES="$COMPOSE_FILES compose.chatterbox-turbo.yaml"
    ENV_VARS+=("AUGMENTUM_TTS_CHATTERBOX_TURBO_URL=http://chatterbox-turbo:8890")
    ok "Chatterbox Turbo added — Kokoro stays as fallback"
    note "350M English model: lower VRAM, faster, [laugh]/[cough]/[chuckle] tags."
    note "Upload voice samples in Settings > Voice."
    add_summary "Text-to-speech" "Chatterbox Turbo + Kokoro fallback"
    ;;
  3)
    COMPOSE_FILES="$COMPOSE_FILES compose.fish-tts.yaml"
    ENV_VARS+=("AUGMENTUM_TTS_FISH_URL=http://fish-tts:8080")
    ok "Fish Speech added — Kokoro stays as fallback"
    warn "Needs ~8 GB VRAM and a HuggingFace token (gated model)."
    note "Accept the license: huggingface.co/fishaudio/openaudio-s1-mini"
    note "Set the HF token in Settings > General after startup if not set above."
    note "48 inline emotion tags, cloning, 13+ languages. Enable"
    note "\"Emotion-aware TTS\" in Settings > Voice for expressive speech."
    add_summary "Text-to-speech" "Fish Speech + Kokoro fallback"
    ;;
  4)
    COMPOSE_FILES="$COMPOSE_FILES compose.qwen-tts.yaml"
    ENV_VARS+=("AUGMENTUM_TTS_QWEN_URL=http://qwen-tts:8880")
    note "Downloading Qwen3 TTS server..."
    if [ -d "$SCRIPT_DIR/services/qwen-tts/.git" ]; then
      git -C "$SCRIPT_DIR/services/qwen-tts" pull --quiet
    else
      git clone --depth 1 https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi.git "$SCRIPT_DIR/services/qwen-tts"
    fi
    ok "Qwen3 TTS added — Kokoro stays as fallback"
    note "Tip: enable \"Emotion-aware TTS\" in Settings > Voice for expressive speech."
    add_summary "Text-to-speech" "Qwen3 TTS + Kokoro fallback"
    ;;
  5)
    COMPOSE_FILES="$COMPOSE_FILES compose.sesame-csm.yaml"
    ENV_VARS+=("AUGMENTUM_TTS_SESAME_CSM_URL=http://sesame-csm:8920")
    ok "Sesame CSM added as the companion's voice"
    warn "Needs a HuggingFace token + acceptance of TWO gated licenses:"
    note "  huggingface.co/sesame/csm-1b"
    note "  huggingface.co/meta-llama/Llama-3.2-1B"
    note "Set the HF token in Settings > General after startup if not set above."
    note "This is a relational voice, not general TTS: prosody conditions on"
    note "the conversation AND how you just sounded. Best for companion chat;"
    note "narration and read-aloud stay on Kokoro. First boot pulls ~3-4 GB."
    note "Pick the voice in Settings > Companion; clone via Settings > Voice."
    add_summary "Text-to-speech" "Sesame CSM (companion) + Kokoro"
    ;;
esac

# ============================================================
# Step 6: Game Streaming (AGSP)
# ============================================================
print_step 6 $TOTAL_STEPS "Game streaming"
note "Browser-streamed open-source games via WebRTC (Luanti today, more later)."
note "Adds ~500 MB of images; ~2 GB RAM + 2 CPUs per active session; NVENC best."

if prompt_yn "Enable game streaming?" "n"; then
  COMPOSE_FILES="$COMPOSE_FILES compose.game-stream.yaml"
  ENV_VARS+=("AUGMENTUM_GAME_STREAM_ENABLED=true")
  ok "Game streaming enabled — build the images with: ./start.sh build"
  add_summary "Game streaming" "on"
else
  ok "Skipped — re-run setup anytime to enable"
  add_summary "Game streaming" "off"
fi

# ============================================================
# Step 7: HTTPS (always-on; informational only)
# ============================================================
print_step 7 $TOTAL_STEPS "HTTPS"
note "On by default: Caddy serves HTTPS on 6443 with a self-signed cert that"
note "auto-regenerates when your LAN IP changes. HTTP stays on 6100 for"
note "loopback. Phones need HTTPS for mic/camera — accept the cert once."
ok "HTTPS on 6443 ${RULE} HTTP on 6100"
add_summary "HTTPS" "on ${RULE} 6443 (self-signed)"

# Caddy is no longer profile-gated; this variable stays for backward
# compatibility with existing .env files but new installs don't need it.
COMPOSE_PROFILES=""

# ============================================================
# Step 8: Save & launch
# ============================================================
print_step 8 $TOTAL_STEPS "Finishing up"

# Generate the engine model mount override if a model directory was given.
# Mirrors setup.bat: hand-edited files (no auto-gen header) are preserved.
ENGINE_OVERRIDE="$SCRIPT_DIR/compose.engine-models.yaml"
if [ -n "${ENGINE_MODEL_MOUNT:-}" ]; then
  OVERRIDE_IS_AUTO=1
  if [ -f "$ENGINE_OVERRIDE" ] && ! head -n1 "$ENGINE_OVERRIDE" | grep -q "^# Generated by setup"; then
    OVERRIDE_IS_AUTO=0
  fi
  if [ "$OVERRIDE_IS_AUTO" -eq 1 ]; then
    {
      echo "# Generated by setup.sh — mounts host model directory"
      echo "services:"
      echo "  augmentum:"
      echo "    volumes:"
      echo "      - $ENGINE_MODEL_MOUNT:/data/host-models:ro"
    } > "$ENGINE_OVERRIDE"
    note "Generated compose.engine-models.yaml for the model mount."
  else
    note "Existing hand-edited compose.engine-models.yaml preserved."
  fi
fi
# The override's existence implies intent, independent of this run's
# answers — skipping the model-dir question must not drop hand-edited
# mounts from the stack.
if [ -f "$ENGINE_OVERRIDE" ]; then
  case " $COMPOSE_FILES " in
    *" compose.engine-models.yaml "*) ;;
    *)
      COMPOSE_FILES="$COMPOSE_FILES compose.engine-models.yaml"
      note "Including existing compose.engine-models.yaml (model mounts)."
      ;;
  esac
fi

# Carry over compose files the wizard doesn't manage (e.g. calling,
# rsshub, hand-added overlays). The wizard only rebuilds what it asked
# about; everything else in the previous conf survives the re-run.
_WIZARD_COMPOSE="compose.yaml compose.dev.yaml compose.gpu.yaml compose.speaches.yaml compose.chatterbox.yaml compose.chatterbox-turbo.yaml compose.fish-tts.yaml compose.qwen-tts.yaml compose.sesame-csm.yaml compose.game-stream.yaml compose.engine-models.yaml compose.classifier.yaml"
for f in ${EXISTING:-}; do
  case " $_WIZARD_COMPOSE " in *" $f "*) continue ;; esac
  case " $COMPOSE_FILES " in *" $f "*) continue ;; esac
  COMPOSE_FILES="$COMPOSE_FILES $f"
  note "Carried over $f from previous configuration."
done

# Local voice/intent classifier — built-in companion resource, default-on.
# Small sidecar the classifier/utility (and, for Gemma, vision) roles resolve
# to; graceful-degrades to the primary chat model if removed. Idempotent.
#
# Model choice (VRAM-aware default; user confirms). Writes the external
# classifier container's -hf model (the working auto-pull path) AND the managed
# "Slot C" defaults so enabling the runtime-switchable slot later picks up the
# same model. Gemma E2B/E4B are vision-capable (their mmproj makes the slot the
# captioner — no separate SmolVLM on GPU); SmolLM2 is text-only (CPU tier).
CLF_DEFAULT=1   # SmolLM2-135M (CPU) unless a capable GPU is detected
if [ -n "$GPU_NAME" ] && [ "$GPU_VRAM_MB" -ge 10000 ]; then
  CLF_DEFAULT=3
elif [ -n "$GPU_NAME" ] && [ "$GPU_VRAM_MB" -ge 5000 ]; then
  CLF_DEFAULT=2
fi
note "Classifier model — the voice/utility/vision workhorse${GPU_NAME:+ (detected ${GPU_VRAM_MB} MB VRAM)}"
prompt_choice "Classifier model" $CLF_DEFAULT \
  "SmolLM2-135M — CPU, text-only (no vision); smallest footprint" \
  "Gemma-4-E2B — GPU, fast + can SEE (vision); ~5 GB VRAM" \
  "Gemma-4-E4B — GPU, best judgment + vision; ~10 GB VRAM"
CLF_GPU=0
case $PROMPT_CHOICE in
  0)
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_HF=bartowski/SmolLM2-135M-Instruct-GGUF:Q8_0")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_NGL=0")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_SAMPLING_TEMPERATURE=0.0")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_SAMPLING_TOP_P=1.0")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_SAMPLING_TOP_K=0")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_SLOT_MODEL=SmolLM2-135M-Instruct-Q8_0")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_SLOT_GPU_LAYERS=0")
    add_summary "Classifier" "SmolLM2-135M (CPU, text-only)"
    ;;
  1)
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_HF=unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_NGL=99")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_CTX=32768")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_SAMPLING_TEMPERATURE=1.0")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_SAMPLING_TOP_P=0.95")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_SAMPLING_TOP_K=64")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_VISION_ARGS=--mmproj-url https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF/resolve/main/mmproj-F16.gguf")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_SLOT_MODEL=gemma-4-E2B-it-qat-UD-Q4_K_XL")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_SLOT_GPU_LAYERS=99")
    CLF_GPU=1
    add_summary "Classifier" "Gemma-4-E2B (GPU, vision)"
    ;;
  2)
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_HF=unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_NGL=99")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_CTX=32768")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_SAMPLING_TEMPERATURE=1.0")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_SAMPLING_TOP_P=0.95")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_SAMPLING_TOP_K=64")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_VISION_ARGS=--mmproj-url https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF/resolve/main/mmproj-F16.gguf")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_SLOT_MODEL=gemma-4-E4B-it-qat-UD-Q4_K_XL")
    ENV_VARS+=("AUGMENTUM_CLASSIFIER_SLOT_GPU_LAYERS=99")
    CLF_GPU=1
    add_summary "Classifier" "Gemma-4-E4B (GPU, vision)"
    ;;
esac

case " $COMPOSE_FILES " in
  *" compose.classifier.yaml "*) ;;
  *) COMPOSE_FILES="$COMPOSE_FILES compose.classifier.yaml" ;;
esac
# GPU classifier → add the CUDA overlay so the container actually offloads.
if [ "$CLF_GPU" -eq 1 ]; then
  case " $COMPOSE_FILES " in
    *" compose.classifier-gpu.yaml "*) ;;
    *) COMPOSE_FILES="$COMPOSE_FILES compose.classifier-gpu.yaml" ;;
  esac
fi

# Save compose file list
echo "$COMPOSE_FILES" > "$CONF_FILE"

# Random-secret generator (256 bits, hex). Tries openssl, then python, then
# /dev/urandom — at least one is present on effectively every Linux/Mac box.
_gen_hex32() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c "import secrets; print(secrets.token_hex(32))"
  elif command -v python >/dev/null 2>&1; then
    python -c "import secrets; print(secrets.token_hex(32))"
  else
    head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'
  fi
}

# Preserve a managed secret across re-runs, or generate a fresh one. Treats
# the documented placeholder as "unset" so dev defaults never survive setup.
#
#   _preserve_or_gen <env_key> <placeholder_to_replace> [generator_fn]
#
# generator_fn defaults to _gen_hex32. Echoes the resulting value.
_preserve_or_gen() {
  local key="$1" placeholder="$2" gen="${3:-_gen_hex32}"
  local value=""
  if [ -f "$ENV_FILE" ]; then
    value=$(grep -E "^${key}=" "$ENV_FILE" 2>/dev/null | head -n1 | cut -d= -f2- || true)
    if [ "$value" = "$placeholder" ]; then
      value=""
    fi
  fi
  if [ -z "$value" ]; then
    value=$("$gen")
  fi
  echo "$value"
}

# SearXNG session secret — must be a stable random value, not the literal
# placeholder. Re-running setup preserves the existing secret so users
# don't lose any searxng preference cookies.
SEARXNG_SECRET=$(_preserve_or_gen SEARXNG_SECRET "change-me-in-production")
ENV_VARS+=("SEARXNG_SECRET=$SEARXNG_SECRET")

# TURN server shared secret (coturn ``--use-auth-secret`` HMAC). The proxy
# (augmentum/calling/turn_credentials.py) mints ephemeral creds against
# this; both sides must agree or every relay attempt is rejected. The
# documented dev default ships in compose.calling.yaml as a fallback so
# `docker compose up` boots without setup, but anyone reachable on the
# coturn port could mint creds against it — must be rotated for any
# deployment outside localhost.
AUGMENTUM_TURN_SECRET=$(_preserve_or_gen \
  AUGMENTUM_TURN_SECRET "augmentum-turn-dev-secret-change-in-env")
ENV_VARS+=("AUGMENTUM_TURN_SECRET=$AUGMENTUM_TURN_SECRET")

# LiveKit API key + secret — the proxy mints room JWTs with these, the
# livekit container validates them. Both sides read from .env; the
# compose.calling.yaml dev defaults are placeholders only.
#
# LIVEKIT_API_KEY is an opaque short identifier (not a secret). Generate
# a stable random key with a clear prefix so it's distinguishable from
# the secret in logs and configs.
_gen_key_id() { printf 'aug_%s' "$(_gen_hex32 | cut -c1-16)"; }
LIVEKIT_API_KEY=$(_preserve_or_gen LIVEKIT_API_KEY "devkey" _gen_key_id)
ENV_VARS+=("LIVEKIT_API_KEY=$LIVEKIT_API_KEY")

LIVEKIT_API_SECRET=$(_preserve_or_gen \
  LIVEKIT_API_SECRET "augmentum-livekit-dev-secret-change-in-env")
ENV_VARS+=("LIVEKIT_API_SECRET=$LIVEKIT_API_SECRET")

# Build COMPOSE_FILE from compose file list (colon-separated for Linux/Mac)
COMPOSE_FILE_VAR=""
for f in $COMPOSE_FILES; do
  if [ -z "$COMPOSE_FILE_VAR" ]; then
    COMPOSE_FILE_VAR="$f"
  else
    COMPOSE_FILE_VAR="$COMPOSE_FILE_VAR:$f"
  fi
done

# Preserve user customizations across re-runs (parity with setup.bat).
# ----------------------------------------------------------------------
# Setup actively manages a fixed set of keys; any OTHER key=value lines in
# the existing .env are user-set — AUGMENTUM_BIND_HOST, extra model dirs,
# provider API keys, etc. Without preservation every re-run silently wipes
# them. A single backup at .env.bak is kept as a safety net.
_MANAGED_KEYS="COMPOSE_FILE COMPOSE_PROFILES SEARXNG_SECRET AUGMENTUM_TURN_SECRET LIVEKIT_API_KEY LIVEKIT_API_SECRET AUGMENTUM_VARIANT AUGMENTUM_DOCKERFILE AUGMENTUM_ENGINE_MANAGED AUGMENTUM_ENGINE_MODEL_DIR AUGMENTUM_DEFAULT_BACKEND AUGMENTUM_OPENAI_BASE_URL AUGMENTUM_OPENAI_API_KEY AUGMENTUM_HUGGINGFACE_TOKEN AUGMENTUM_STT_PROVIDER_URL AUGMENTUM_STT_DEFAULT_MODEL AUGMENTUM_STT_MODEL AUGMENTUM_VOICE_MOONSHINE_ENABLED AUGMENTUM_TTS_CHATTERBOX_URL AUGMENTUM_CHATTERBOX_DEVICE AUGMENTUM_TTS_CHATTERBOX_TURBO_URL AUGMENTUM_TTS_FISH_URL AUGMENTUM_TTS_QWEN_URL AUGMENTUM_TTS_SESAME_CSM_URL AUGMENTUM_GAME_STREAM_ENABLED AUGMENTUM_CLASSIFIER_HF AUGMENTUM_CLASSIFIER_NGL AUGMENTUM_CLASSIFIER_CTX AUGMENTUM_CLASSIFIER_SAMPLING_TEMPERATURE AUGMENTUM_CLASSIFIER_SAMPLING_TOP_P AUGMENTUM_CLASSIFIER_SAMPLING_TOP_K AUGMENTUM_CLASSIFIER_VISION_ARGS AUGMENTUM_CLASSIFIER_SLOT_MODEL AUGMENTUM_CLASSIFIER_SLOT_GPU_LAYERS"
USER_OVERRIDES=""
if [ -f "$ENV_FILE" ]; then
  cp -f "$ENV_FILE" "$ENV_FILE.bak"
  _managed_pattern="^($(echo "$_MANAGED_KEYS" | tr ' ' '|'))="
  # Keep user KEY= lines AND their comment/blank lines — people document
  # their .env and that documentation must survive re-runs. Only the
  # managed keys and setup's own marker comments are dropped.
  USER_OVERRIDES=$(grep -Ev "$_managed_pattern" "$ENV_FILE" \
    | grep -Ev '^# Generated by Augmentum setup|^# User customizations preserved from previous' || true)
fi

# Always write .env (COMPOSE_FILE ensures `docker compose up` works without start.sh)
{
  echo "# Generated by Augmentum setup — $(date -Iseconds 2>/dev/null || date)"
  echo "COMPOSE_FILE=$COMPOSE_FILE_VAR"
  for var in "${ENV_VARS[@]}"; do
    echo "$var"
  done
  if [ -n "$USER_OVERRIDES" ]; then
    echo ""
    echo "# User customizations preserved from previous .env"
    printf '%s\n' "$USER_OVERRIDES"
  fi
} > "$ENV_FILE"

if [ -n "$USER_OVERRIDES" ]; then
  _n_overrides=$(printf '%s\n' "$USER_OVERRIDES" | wc -l | tr -d ' ')
  note "Preserved ${_n_overrides} user customization(s); previous .env backed up to .env.bak"
fi
add_summary "Saved" ".augmentum.conf ${RULE} .env"

# --- Summary panel ---------------------------------------------------------

echo ""
hr 50
echo "  ${BOLD}${GREEN}${CHECK} Setup complete${RESET}"
echo ""
for i in "${!SUMMARY_KEYS[@]}"; do
  printf "    %s%-16s%s %s\n" "$DIM" "${SUMMARY_KEYS[$i]}" "$RESET" "${SUMMARY_VALS[$i]}"
done
hr 50
echo ""
echo "  ${BOLD}Open${RESET}            ${CYAN}https://localhost:6443${RESET}"
echo "  ${BOLD}From your LAN${RESET}   ${CYAN}https://<your-lan-ip>:6443${RESET} ${DIM}(accept cert once per device)${RESET}"
if [ -n "$STT_WEBUI" ]; then
  echo "  ${BOLD}STT WebUI${RESET}       ${CYAN}${STT_WEBUI}${RESET}"
fi
if [ -n "$TTS_WEBUI" ]; then
  echo "  ${BOLD}TTS WebUI${RESET}       ${CYAN}${TTS_WEBUI}${RESET}"
fi
echo ""
echo "  ${DIM}Commands: ./start.sh (start) ${RULE} ./start.sh -d (background) ${RULE} ./start.sh down (stop)${RESET}"
echo "  ${DIM}          ./start.sh logs (logs) ${RULE} ./setup.sh (reconfigure)${RESET}"
echo ""

if prompt_yn "Start Augmentum now?" "y"; then
  echo ""
  exec "$SCRIPT_DIR/start.sh"
else
  echo ""
  echo "  Run ${CYAN}./start.sh${RESET} when you're ready."
  echo ""
fi
