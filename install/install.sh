#!/usr/bin/env bash
set -euo pipefail

# Augmentum universal installer (Linux + macOS, any CPU arch).
#
# ONE command, any Unix machine:
#   curl -fsSL https://raw.githubusercontent.com/AugmentumHQ/Augmentum/main/install/install.sh | bash
#
# It detects your OS and CPU, installs a container runtime if one is missing
# (Docker Engine on Linux, Colima on macOS — Rosetta-accelerated on Apple
# Silicon), fetches the pull-only compose.yaml into ./augmentum, enables the
# bundled Caddy HTTPS proxy, pulls the published CPU image from GHCR, and
# starts everything. For the GPU variant or to choose optional services, run
# ./setup.sh from a git clone. (Windows: use install.ps1.)

RAW_BASE="https://raw.githubusercontent.com/AugmentumHQ/Augmentum/main"

OS="$(uname -s)"
ARCH="$(uname -m)"

echo "=== Augmentum Installer ==="
echo "    OS: $OS   CPU: $ARCH"
echo ""

# ==========================================================================
# 1. Container runtime — bootstrap per-OS if Docker isn't already usable.
# ==========================================================================
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    echo "[OK] Docker detected"
    docker --version
elif [ "$OS" = "Darwin" ]; then
    # ---- macOS: Homebrew + Colima ----
    if ! command -v brew &>/dev/null; then
        echo "[!] Homebrew not found. Installing..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    echo "[!] Docker not found. Installing via Colima..."
    brew install docker docker-compose colima
    echo "Starting Colima VM (this may take a minute)..."
    # The published images are linux/amd64. On Apple Silicon (arm64), run a
    # native-amd64 Colima VM accelerated by Rosetta so the amd64 layers are
    # fast; on Intel a plain VM is already amd64. Falls back to a plain VM if
    # the vz/Rosetta backend isn't available (older macOS / Colima).
    if [ "$ARCH" = "arm64" ]; then
        echo "    Apple Silicon detected — using Rosetta-accelerated amd64 VM."
        colima start --arch x86_64 --vm-type vz --vz-rosetta \
            --cpu 4 --memory 8 --disk 60 \
        || colima start --arch x86_64 --cpu 4 --memory 8 --disk 60
    else
        colima start --cpu 4 --memory 8 --disk 60
    fi
    echo "[OK] Docker ready via Colima"
else
    # ---- Linux: Docker Engine via get.docker.com ----
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        echo "[!] ARM host detected ($ARCH). The published images are amd64, so"
        echo "    they run under QEMU emulation here (works, but slower). If a"
        echo "    pull fails with an 'exec format' error, enable QEMU:"
        echo "      docker run --privileged --rm tonistiigi/binfmt --install amd64"
        echo "    A native arm64 build is a planned fast-follow."
        echo ""
    fi
    echo "[!] Docker not found. Installing Docker Engine..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo ""
    echo "[!] Docker installed. Log out and back in (or run 'newgrp docker')"
    echo "    for the group change to take effect, then re-run this installer."
    exit 0
fi

# ==========================================================================
# 2. Docker Compose plugin.
# ==========================================================================
if docker compose version &>/dev/null 2>&1; then
    echo "[OK] Docker Compose detected"
else
    echo "[!] Docker Compose plugin not found."
    if [ "$OS" = "Darwin" ]; then
        echo "    brew install docker-compose"
    else
        echo "    Install it: sudo apt-get install docker-compose-plugin"
        echo "    (or follow https://docs.docker.com/compose/install/)"
    fi
    exit 1
fi

# ==========================================================================
# 3. Project directory — reuse a compose.yaml here, else fetch pull-only one.
# ==========================================================================
if [ -f compose.yaml ]; then
    echo "[OK] Using compose.yaml in $(pwd)"
else
    TARGET_DIR="${AUGMENTUM_DIR:-augmentum}"
    echo "Setting up Augmentum in ./$TARGET_DIR ..."
    mkdir -p "$TARGET_DIR"
    cd "$TARGET_DIR"
    curl -fsSL -o compose.yaml "$RAW_BASE/compose.yaml"
    if [ ! -f .env ]; then
        cat > .env <<'EOF'
AUGMENTUM_VARIANT=cpu
# Run the bundled Caddy reverse proxy so the UI is served over HTTPS on :6443
# (self-signed cert — accept the browser warning once). Plain HTTP stays on
# :6100. HTTPS is required for microphone/camera access from other devices on
# a LAN. Set this empty to disable Caddy and use http://localhost:6100/ui only.
COMPOSE_PROFILES=https
EOF
    fi
fi

# ==========================================================================
# 4. Pull + start.
# ==========================================================================
echo ""
echo "Pulling Augmentum CPU variant from GHCR..."
# Retry the pull: Docker Hub's CDN (cloudfront) intermittently throws transient
# TLS handshake timeouts on the public base images (caddy/searxng/etc.). A retry
# almost always clears it, and Docker resumes already-downloaded layers — so a
# first-time user never sees a one-off network flake as an "install failed".
pull_ok=false
for attempt in 1 2 3 4 5; do
    if docker compose pull; then pull_ok=true; break; fi
    if [ "$attempt" -lt 5 ]; then
        wait_s=$((attempt * 5))
        echo "  Pull attempt $attempt didn't finish (often a transient Docker Hub CDN hiccup)."
        echo "  Retrying in ${wait_s}s... (Docker keeps the layers it already got)"
        sleep "$wait_s"
    fi
done
if [ "$pull_ok" != true ]; then
    echo ""
    echo "  Pull still failing after 5 attempts. This is almost always a network/CDN"
    echo "  issue upstream, not Augmentum. Re-run 'docker compose pull' (it resumes)."
    echo "  On WSL2, a wedged network stack can cause this — try 'wsl --shutdown'"
    echo "  (from PowerShell), restart Docker, then re-run the pull."
    exit 1
fi
echo ""
echo "Starting services..."
docker compose up -d

echo ""
echo "=== Augmentum is starting ==="
echo "Open https://localhost:6443 in your browser — it uses a self-signed"
echo "certificate, so accept the warning once. Plain HTTP is also available at"
echo "http://localhost:6100/ui."
echo "The first user to register becomes the admin — pick a username and password."
echo ""
echo "If you set AUGMENTUM_BIND_HOST=0.0.0.0 to reach Augmentum from another"
echo "device, register that admin account BEFORE opening the UI elsewhere."
echo ""
if [ "$OS" = "Darwin" ]; then
    echo "Want to choose optional services (TTS, STT, ...)? See the README, or"
    echo "run ./setup.sh from a git clone. (GPU passthrough isn't available on macOS.)"
else
    echo "Want the GPU variant or to choose optional services (TTS, image gen,"
    echo "game streaming, ...)? See the README, or run ./setup.sh from a git clone."
fi
echo ""
echo "Commands (run from $(pwd)):"
echo "  docker compose logs -f    # Watch logs"
echo "  docker compose down       # Stop"
echo "  docker compose up -d      # Start"
echo "  docker compose pull       # Update to the latest image"
if [ "$OS" = "Darwin" ]; then
    echo "  colima stop               # Stop the Colima VM (frees CPU/RAM)"
fi
