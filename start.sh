#!/usr/bin/env bash
# ============================================================================
# Augmentum Start — Reads .augmentum.conf and runs docker compose
#
# Usage:
#   ./start.sh              Start (foreground)
#   ./start.sh -d           Start (detached/background)
#   ./start.sh down         Stop services
#   ./start.sh logs         Tail logs
#   ./start.sh logs -f      Follow logs
#   ./start.sh ps           Show running services
#   ./start.sh restart      Restart services
#   ./start.sh build        Rebuild containers
#   ./start.sh <any>        Pass any docker compose subcommand
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONF_FILE="$SCRIPT_DIR/.augmentum.conf"

if [ ! -f "$CONF_FILE" ]; then
  echo ""
  echo "  No configuration found. Run setup first:"
  echo ""
  echo "    ./setup.sh    (Linux/macOS/Git Bash)"
  echo "    setup.bat     (Windows CMD)"
  echo ""
  exit 1
fi

# Read compose file list from config
COMPOSE_FILES=$(cat "$CONF_FILE")

# Build -f flags
COMPOSE_FLAGS=""
for f in $COMPOSE_FILES; do
  COMPOSE_FLAGS="$COMPOSE_FLAGS -f $f"
done

cd "$SCRIPT_DIR"

# Keep the raw-compose entrypoint in sync: regenerate .env's COMPOSE_FILE (+
# COMPOSE_PATH_SEPARATOR) from .augmentum.conf so `docker compose up` resolves
# the SAME overlay set this script does. Non-fatal — this script uses the -f
# flags built above regardless; the sync only serves the bare-compose path.
if command -v python3 >/dev/null 2>&1; then _PY=python3
elif command -v python >/dev/null 2>&1; then _PY=python
else _PY=""; fi
if [ -n "$_PY" ]; then
  "$_PY" "$SCRIPT_DIR/scripts/bootstrap/sync_compose_env.py" || echo "[start] compose-env sync skipped (non-fatal)"
fi

# Detect the host's RFC1918 IPv4 addresses so Caddy can bake them into
# the self-signed cert's SAN list. Operators who set
# AUGMENTUM_TLS_EXTRA_SANS in .env (or as a shell env) keep full control —
# we only auto-populate when it's unset. Without this, a fresh install
# accessed from a phone or another LAN box would hit a cert name-mismatch
# wall because the cert only covers localhost/0.0.0.0 by default.
#
# Output shape (suitable for AUGMENTUM_TLS_EXTRA_SANS): comma-separated
# "IP:a.b.c.d" tokens. Empty if no usable address found.
detect_host_lan_sans() {
  local ips=""
  if command -v hostname >/dev/null 2>&1; then
    # `hostname -I` is the Linux happy path; one space-separated list of
    # every assigned address. macOS' hostname doesn't accept -I, so guard.
    ips=$(hostname -I 2>/dev/null || true)
  fi
  if [ -z "$ips" ] && command -v ifconfig >/dev/null 2>&1; then
    # macOS fallback. -a covers all interfaces; awk pulls every `inet`
    # IPv4 after stripping the netmask.
    ips=$(ifconfig -a 2>/dev/null | awk '/inet /{print $2}' || true)
  fi
  if [ -z "$ips" ] && command -v ip >/dev/null 2>&1; then
    # ip-utils fallback when hostname is unavailable (Alpine-style boxes).
    ips=$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 || true)
  fi
  local sans=""
  for ip in $ips; do
    # Docker bridge gateway IPs (172.17.0.1 default; 172.18-31.0.1 custom
    # networks) appear in `hostname -I` on Linux when Docker is running,
    # but they're container-only — no device outside the box ever sees
    # them. Including them in the cert just bloats it. Filter the
    # ``172.X.0.1`` shape specifically; non-gateway 172.16-31.x.x IPs
    # (e.g. a real corporate LAN in 172.20/12) still pass.
    case "$ip" in
      172.1[7-9].0.1|172.2[0-9].0.1|172.3[01].0.1)
        continue
        ;;
    esac
    case "$ip" in
      # RFC1918 + CGNAT-ish Tailscale 100.64/10 range. Loopback / link-local
      # excluded — the cert already covers 127.0.0.1, and 169.254.x.x is
      # never reachable from another device.
      10.*|192.168.*|172.16.*|172.17.*|172.18.*|172.19.*|172.2[0-9].*|172.3[01].*|100.6[4-9].*|100.[7-9][0-9].*|100.1[0-1][0-9].*|100.12[0-7].*)
        if [ -n "$sans" ]; then
          sans="$sans,IP:$ip"
        else
          sans="IP:$ip"
        fi
        ;;
    esac
  done
  printf '%s' "$sans"
}

# Auto-populate AUGMENTUM_TLS_EXTRA_SANS if the operator hasn't set it
# via .env or shell env. start.sh exporting this means the value reaches
# docker compose (which reads it for the caddy service environment).
if [ -z "${AUGMENTUM_TLS_EXTRA_SANS:-}" ]; then
  # Read .env so an explicit value there overrides our auto-detection.
  if [ -f .env ] && grep -qE '^\s*AUGMENTUM_TLS_EXTRA_SANS=' .env 2>/dev/null; then
    : # .env will be picked up by docker compose; leave it alone
  else
    AUTO_SANS=$(detect_host_lan_sans)
    if [ -n "$AUTO_SANS" ]; then
      export AUGMENTUM_TLS_EXTRA_SANS="$AUTO_SANS"
      echo "  TLS SANs auto-detected: $AUTO_SANS"
    fi
  fi
fi

# Auto-detect the node's Tailscale MagicDNS name (<node>.<tailnet>.ts.net). This
# gives the app a STABLE tailnet URL and — once the operator enables Funnel — a
# durable public guest address, instead of falling to an anonymous cloudflared
# tunnel whose URL changes every restart. Zero-config; skipped when Tailscale or
# a JSON parser isn't available. Operator override via .env / shell wins.
if [ -z "${AUGMENTUM_TAILNET_HOSTNAME:-}" ] && command -v tailscale >/dev/null 2>&1 && [ -n "$_PY" ]; then
  if [ -f .env ] && grep -qE '^\s*AUGMENTUM_TAILNET_HOSTNAME=' .env 2>/dev/null; then
    : # .env value reaches docker compose; leave it alone
  else
    TS_NAME=$(tailscale status --json 2>/dev/null | "$_PY" -c 'import sys,json; d=json.load(sys.stdin); print((d.get("Self",{}).get("DNSName") or "").rstrip("."))' 2>/dev/null)
    if [ -n "$TS_NAME" ]; then
      export AUGMENTUM_TAILNET_HOSTNAME="$TS_NAME"
      echo "  Tailscale name: $TS_NAME (tailnet URL; enable Funnel for durable guest access)"
    fi
  fi
fi

# Export the agent-browser pin so compose.browser.yaml's build arg gets
# the REAL pinned version, not the Dockerfile default (verified-build-
# inputs discipline — same reason the llama-server build passes its
# version explicitly). No-op when the pin file is absent.
if [ -f "$SCRIPT_DIR/AGENT_BROWSER_VERSION" ]; then
  export AGENT_BROWSER_VERSION="$(tr -d '[:space:]' < "$SCRIPT_DIR/AGENT_BROWSER_VERSION")"
fi

# Show URLs
show_urls() {
  echo "  HTTP:  http://localhost:6100"
  echo "  HTTPS: https://localhost:6443"
  # Surface each useful interface with its kind so the operator knows
  # which URL to hand a phone (LAN) vs. an away-from-home laptop
  # (Tailscale). Iterates the SAN list — already filtered to LAN +
  # Tailscale by detect_host_lan_sans — and groups by RFC1918 range.
  if [ -n "${AUGMENTUM_TLS_EXTRA_SANS:-}" ]; then
    local token kind ip
    # Replace commas with spaces for the for-loop split.
    for token in $(printf '%s' "$AUGMENTUM_TLS_EXTRA_SANS" | tr ',' ' '); do
      case "$token" in
        IP:*) ip="${token#IP:}" ;;
        *) continue ;;
      esac
      case "$ip" in
        10.*|192.168.*|172.*) kind="LAN" ;;
        100.6[4-9].*|100.[7-9][0-9].*|100.1[0-1][0-9].*|100.12[0-7].*) kind="Tailscale" ;;
        *) kind="" ;;
      esac
      if [ -n "$kind" ]; then
        echo "  HTTPS: https://$ip:6443 ($kind — accept self-signed cert once)"
      fi
    done
  fi
}

# Provision augmentum-llama-server when a local build of augmentum is in
# play (compose.dev.yaml has a build: directive that COPYs the binary out
# of FROM augmentum-llama-server:latest). Production runs that pull the
# prebuilt augmentum image from GHCR don't need this — short-circuit.
ensure_llama_server_image() {
  if docker image inspect augmentum-llama-server >/dev/null 2>&1; then
    return 0
  fi
  if ! grep -q "compose.dev.yaml" "$CONF_FILE" 2>/dev/null; then
    return 0
  fi
  if [ ! -f "$SCRIPT_DIR/LLAMA_SERVER_VERSION" ]; then
    return 0
  fi
  local llama_ver
  llama_ver=$(tr -d '[:space:]' < "$SCRIPT_DIR/LLAMA_SERVER_VERSION")
  # Prefer a LAN registry (Home Main) when configured — skips the GHCR
  # round-trip and the local CUDA compile fallback entirely.
  local aug_registry="${AUGMENTUM_REGISTRY:-}"
  if [ -z "$aug_registry" ] && [ -f "$SCRIPT_DIR/.env" ]; then
    aug_registry="$(grep -E '^AUGMENTUM_REGISTRY=' "$SCRIPT_DIR/.env" | tail -1 | cut -d= -f2- | tr -d '\r')"
  fi
  if [ -n "$aug_registry" ]; then
    echo "  Fetching llama-server $llama_ver from $aug_registry..."
    if docker pull "$aug_registry/augmentumhq/augmentum-llama-server:$llama_ver" >/dev/null 2>&1; then
      docker tag "$aug_registry/augmentumhq/augmentum-llama-server:$llama_ver" augmentum-llama-server:latest >/dev/null 2>&1
      echo "    Pulled augmentum-llama-server:latest from $aug_registry."
      return 0
    fi
    echo "    Local registry pull failed — trying GHCR..."
  fi
  echo "  Fetching llama-server $llama_ver from GHCR..."
  if docker pull "ghcr.io/augmentumhq/augmentum-llama-server:$llama_ver" >/dev/null 2>&1; then
    docker tag "ghcr.io/augmentumhq/augmentum-llama-server:$llama_ver" augmentum-llama-server:latest >/dev/null 2>&1
    echo "    Pulled augmentum-llama-server:latest from GHCR."
    return 0
  fi
  if [ -f "$SCRIPT_DIR/Dockerfile.llama-server" ]; then
    echo "    GHCR pull failed — falling back to local CUDA compile (~30-50 min)..."
    if ! docker build -t augmentum-llama-server \
          --build-arg "LLAMA_CPP_VERSION=$llama_ver" \
          --progress=plain \
          -f Dockerfile.llama-server .; then
      echo "  [warning] llama-server build failed — built-in engine will be unavailable"
    fi
  else
    echo "  [warning] llama-server image unavailable — built-in engine will be unavailable"
  fi
}

# Background wrapper for the llama-server builder image (2026-07-02
# boot-latency work). The RUNNING stack never needs this image — the
# binary is baked into the app image; it only feeds the next
# `start.sh build`. When the image exists this is a ~100ms no-op; when
# it's missing (post-prune, fresh dev checkout) the GHCR pull or the
# 30-50 min CUDA compile used to stand between the user and a stack
# that didn't need it. Now it runs detached; progress lands in the log.
# The `build` path below still calls ensure_llama_server_image inline —
# that path genuinely needs the image before building the app image.
ensure_llama_server_async() {
  if docker image inspect augmentum-llama-server >/dev/null 2>&1; then
    return 0
  fi
  echo "  llama-server builder image missing -- provisioning in background"
  echo "  (log: /tmp/augmentum-llama-ensure.log; only needed by 'start.sh build')"
  ( ensure_llama_server_image > /tmp/augmentum-llama-ensure.log 2>&1 & )
}

# Keep the game-stream images referenced by a running container so a
# routine `docker image prune -a` / Docker Desktop "Clean up" can't sweep
# them as unused -- they're spawned on demand, so between sessions nothing
# holds them and they look like garbage. Also warns when an image isn't
# built, so that surfaces at startup instead of as a per-title 404 at
# launch. Non-fatal by construction: the script always exits 0 and this
# must never block boot.
ensure_gs_anchors() {
  if command -v python3 >/dev/null 2>&1; then _ANCHOR_PY=python3
  elif command -v python >/dev/null 2>&1; then _ANCHOR_PY=python
  else return 0; fi
  "$_ANCHOR_PY" "$SCRIPT_DIR/scripts/bootstrap/ensure_game_stream_anchors.py" || true
}

# Default action: up (no rebuild unless explicitly requested)
if [ $# -eq 0 ]; then
  ensure_llama_server_async
  echo "  Starting Augmentum..."
  echo "  Config: $COMPOSE_FILES"
  show_urls
  echo ""
  ensure_gs_anchors
  exec docker compose $COMPOSE_FLAGS up
fi

# -d shorthand
if [ "$1" = "-d" ]; then
  ensure_llama_server_async
  echo "  Starting Augmentum (detached)..."
  echo "  Config: $COMPOSE_FILES"
  show_urls
  echo ""
  ensure_gs_anchors
  exec docker compose $COMPOSE_FLAGS up -d
fi

# build: rebuild then start
if [ "$1" = "build" ]; then
  ensure_llama_server_image
  echo "  Building and starting Augmentum..."
  echo "  Config: $COMPOSE_FILES"
  echo ""
  # Build the workspace profile tags (coder mode containers). Multi-stage
  # Dockerfile produces one tag per profile so workspaces get the right
  # prebake without re-installing on every create. See
  # docs/superpowers/specs/2026-06-02-tooling-profile-system-v2.md.
  if [ -f Dockerfile.workspace ]; then
    echo "  Building workspace profile tags (standard, power, browser)..."
    echo "  Total disk usage ~1.4 GB (profiles share layers)."
    if ! docker build -t augmentum-workspace:standard --target standard -f Dockerfile.workspace .; then
      echo "  [warning] standard profile build failed — coder mode will use ubuntu:24.04 fallback"
    fi
    if ! docker build -t augmentum-workspace:power --target power -f Dockerfile.workspace .; then
      echo "  [warning] power profile build failed — workspaces with profile=power will fall back to :standard + runtime install"
    fi
    if ! docker build -t augmentum-workspace:browser --target browser -f Dockerfile.workspace .; then
      echo "  [warning] browser profile build failed — workspaces with profile=browser will fall back to :power + runtime install"
    fi
    # Pentest profile is opt-in: only built when both pin files exist AND
    # METASPLOIT_SHA256 has been populated by scripts/upgrade_msf.sh.
    # The placeholder "UNPINNED" sha would fail the sha-check inside the
    # Dockerfile stage anyway — skipping with a clear message is friendlier
    # than letting the build error out mid-run.
    if [ -f METASPLOIT_VERSION ] && [ -f METASPLOIT_SHA256 ]; then
      msf_version="$(tr -d '\n\r' < METASPLOIT_VERSION)"
      msf_sha="$(tr -d '\n\r' < METASPLOIT_SHA256)"
      if [ -n "$msf_version" ] && [ -n "$msf_sha" ] && [ "$msf_sha" != "UNPINNED" ]; then
        echo "  Building pentest profile (Metasploit ${msf_version}, ~4 GB)..."
        if ! docker build \
                -t augmentum-workspace:pentest \
                --target pentest \
                --build-arg "METASPLOIT_VERSION=${msf_version}" \
                --build-arg "METASPLOIT_SHA256=${msf_sha}" \
                -f Dockerfile.workspace .; then
          echo "  [warning] pentest profile build failed — workspaces with profile=pentest will fall back to :power + runtime install"
        fi
      else
        echo "  [info] pentest profile skipped — METASPLOIT_SHA256 is unset/placeholder."
        echo "          Run scripts/upgrade_msf.sh to populate the pin, then rerun start.sh build."
      fi
    fi
    # Backward compat: keep the v1 unversioned tag pointing at browser
    # (the default profile) so callers that hardcoded "augmentum-workspace"
    # keep working without code change.
    docker tag augmentum-workspace:browser augmentum-workspace:latest 2>/dev/null || echo "  [info] could not tag :latest"
    docker tag augmentum-workspace:browser augmentum-workspace 2>/dev/null || echo "  [info] could not tag generic alias"
  fi
  echo ""
  # Build the game-stream images when game streaming is enabled in .augmentum.conf
  if grep -q "compose.game-stream.yaml" "$CONF_FILE" 2>/dev/null; then
    echo "  Building game-stream base image..."
    if ! docker build -t augmentum-game-stream-base \
            -f services/game-stream/Dockerfile.base .; then
      echo "  [warning] game-stream base build failed — streaming will be unavailable"
    else
      echo "  Building game-stream Luanti image..."
      if ! docker build -t augmentum-game-stream-luanti \
              -f services/game-stream/Dockerfile.luanti .; then
        echo "  [warning] game-stream Luanti build failed — streaming will be unavailable"
      fi
      # Emulator-streamed image is in compose profile build-only,
      # so `docker compose up --build` skips it. Build explicitly.
      echo "  Building game-stream emulator-streamed image (Dolphin + PCSX2)..."
      if ! docker build -t augmentum-game-stream-emulator-streamed \
              -f services/game-stream/Dockerfile.emulator-streamed .; then
        echo "  [warning] game-stream emulator-streamed build failed — GameCube/Wii/PS2 will be unavailable"
      fi
      # Browser-stream image. Also build-only in compose, so `up --build`
      # skips it too — without this cast-to-TV of web surfaces has no
      # image on any fresh install and fails with a 404 at cast time.
      echo "  Building stream-browser image (cast-to-TV of web surfaces)..."
      if ! docker build -t augmentum-stream-browser \
              -f services/game-stream/Dockerfile.browser .; then
        echo "  [warning] stream-browser build failed — casting web surfaces will be unavailable"
      fi
    fi
  fi
  # Re-anchor AFTER the builds: a rebuilt image has a new ID, and an
  # anchor still pinning the OLD one leaves the new image unprotected.
  ensure_gs_anchors
  exec docker compose $COMPOSE_FLAGS up --build
fi

# Pass through any other docker compose subcommand
exec docker compose $COMPOSE_FLAGS "$@"
