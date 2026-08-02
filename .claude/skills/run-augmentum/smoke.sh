#!/usr/bin/env bash
# Augmentum smoke driver.
#
# Brings the compose stack up (if not already) and verifies the proxy is
# answering on http://localhost:6100. Exits non-zero on any failure so
# CI / agents can branch on `$?`.
#
# Usage:
#   ./smoke.sh                # full smoke (boot + probes + screenshot)
#   ./smoke.sh --no-boot      # skip boot, just probe what's running
#   ./smoke.sh --no-shot      # skip the chrome screenshot
#   ./smoke.sh --keep         # leave stack running after success (default: leave running either way)
#
# Artifacts:
#   /tmp/augmentum-smoke.log              boot + probe transcript
#   <repo>/tmp/run-skill/ui-root.png      headless screenshot of /ui/
#
# Notes:
#   - Designed to run from git-bash on Windows (the dev host) or Linux.
#   - On Windows, Chrome's --screenshot path must be Windows-style
#     ("C:\\..."), so the script translates the output path before invoking
#     chrome.exe. cygpath is used when available; pure-bash fallback otherwise.
#   - The smoke uses unauthenticated endpoints only: /api/version,
#     /api/auth/status, and / (Ollama-compat shim). /api/health is gated
#     by auth and is NOT a smoke target.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SHOT_DIR="$REPO_ROOT/tmp/run-skill"
SHOT_PATH="$SHOT_DIR/ui-root.png"
LOG="/tmp/augmentum-smoke.log"
HTTP_BASE="${AUGMENTUM_HTTP_BASE:-http://localhost:6100}"

BOOT=1
SHOT=1
for arg in "$@"; do
  case "$arg" in
    --no-boot) BOOT=0 ;;
    --no-shot) SHOT=0 ;;
    --keep) ;;  # accepted for symmetry; smoke never tears the stack down
    -h|--help)
      sed -n '2,25p' "$0"
      exit 0
      ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$SHOT_DIR"
: > "$LOG"

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

# ---- Step 1: optionally boot the stack -----------------------------------

container_status() {
  docker inspect augmentum-augmentum-1 --format '{{.State.Health.Status}}' 2>/dev/null || echo "absent"
}

if [ "$BOOT" = "1" ]; then
  status=$(container_status)
  if [ "$status" = "healthy" ]; then
    log "augmentum-augmentum-1 already healthy — skipping boot"
  else
    log "augmentum-augmentum-1 status=$status — bringing stack up (detached)"
    # start.sh / start.bat both honour .augmentum.conf and run
    # `docker compose -f ... up -d`. Use the shell variant from git-bash.
    if [ -x "$REPO_ROOT/start.sh" ]; then
      (cd "$REPO_ROOT" && ./start.sh -d) | tee -a "$LOG"
    else
      log "start.sh not executable — falling back to: bash start.sh -d"
      (cd "$REPO_ROOT" && bash start.sh -d) | tee -a "$LOG"
    fi
    # Wait up to ~120s for the healthcheck to flip to healthy. The
    # container's own /api/version probe (Docker HEALTHCHECK in
    # Dockerfile.gpu) is what we're polling here.
    for _ in $(seq 1 60); do
      [ "$(container_status)" = "healthy" ] && break
      sleep 2
    done
    final=$(container_status)
    [ "$final" = "healthy" ] || { log "FAIL: container never reached healthy (status=$final)"; exit 1; }
    log "container healthy"
  fi
fi

# ---- Step 2: unauthenticated HTTP probes ---------------------------------

probe() {
  local path="$1" expect_substr="$2" code
  body=$(curl -sf -o /tmp/aug-probe-body -w "%{http_code}" "$HTTP_BASE$path" || true)
  code="$body"
  payload=$(head -c 200 /tmp/aug-probe-body | tr -d '\r\n')
  if [ "$code" != "200" ]; then
    log "FAIL: $path -> HTTP $code   body: $payload"
    return 1
  fi
  case "$payload" in
    *"$expect_substr"*)
      log "ok:   $path -> 200   body: $payload"
      ;;
    *)
      log "FAIL: $path -> 200 but missing '$expect_substr'   body: $payload"
      return 1
      ;;
  esac
}

probe "/api/version"     '"version"'        || exit 1
probe "/api/auth/status" '"setup_required"' || exit 1
probe "/"                'Ollama'           || exit 1   # /v1-compat shim

# ---- Step 3: chrome headless screenshot ---------------------------------

if [ "$SHOT" = "1" ]; then
  CHROME=""
  for candidate in \
      "/c/Program Files/Google/Chrome/Application/chrome.exe" \
      "/c/Program Files (x86)/Google/Chrome/Application/chrome.exe" \
      "$(command -v chromium 2>/dev/null || true)" \
      "$(command -v chromium-browser 2>/dev/null || true)" \
      "$(command -v google-chrome 2>/dev/null || true)"; do
    [ -n "$candidate" ] && [ -x "$candidate" ] && CHROME="$candidate" && break
  done
  if [ -z "$CHROME" ]; then
    log "WARN: no chrome/chromium found — skipping screenshot"
  else
    # chrome.exe writes the screenshot path verbatim. On Windows, give it a
    # Windows-style absolute path so the file actually lands on disk.
    win_path="$SHOT_PATH"
    if command -v cygpath >/dev/null 2>&1; then
      win_path=$(cygpath -w "$SHOT_PATH")
    fi
    log "screenshot: $CHROME -> $win_path"
    "$CHROME" --headless --disable-gpu --no-sandbox --hide-scrollbars \
      --window-size=1280,800 \
      --virtual-time-budget=8000 \
      "--screenshot=$win_path" \
      "$HTTP_BASE/ui/" >>"$LOG" 2>&1 || true
    if [ -s "$SHOT_PATH" ]; then
      log "ok:   screenshot -> $SHOT_PATH ($(wc -c <"$SHOT_PATH") bytes)"
    else
      log "FAIL: screenshot empty or missing at $SHOT_PATH"
      exit 1
    fi
  fi
fi

log "smoke PASS  (log: $LOG, shot: $SHOT_PATH)"
