# Augmentum one-line installer for Windows (WSL2 + Docker Engine).
#
# Run in an elevated PowerShell (Administrator):
#   irm https://raw.githubusercontent.com/AugmentumHQ/Augmentum/main/install/install.ps1 | iex
#
# Bootstraps WSL2 + an Ubuntu distro + Docker Engine if missing, fetches the
# pull-only compose.yaml into .\augmentum, enables the bundled Caddy HTTPS
# proxy, pulls the published CPU image from GHCR, and starts everything (inside
# WSL — Docker Engine runs there).
#
# For the GPU variant or to choose which optional services run, see the README
# or run setup.bat from a git clone of the repo.

# NOT "Stop": this script drives external tools (wsl, docker) that legitimately
# exit nonzero (e.g. `command -v docker` before Docker is installed) and print
# benign warnings to stderr (WSL's "Processing /etc/fstab with mount -a failed").
# Under "Stop", PowerShell turns those into fatal NativeCommandErrors and aborts
# the install. We check results explicitly instead ($LASTEXITCODE, Test-Path,
# variable contents); the one place a hard failure matters (the compose.yaml
# download) gets an explicit -ErrorAction Stop below.
$ErrorActionPreference = "Continue"
# PS 7.4+ also throws on nonzero native exit codes unless this is disabled.
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction Ignore) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$RawBase = "https://raw.githubusercontent.com/AugmentumHQ/Augmentum/main"

Write-Host "=== Augmentum Installer (Windows) ===" -ForegroundColor Cyan
Write-Host ""

# --- WSL2 -----------------------------------------------------------------
$wslInstalled = $false
try {
    wsl --status 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { $wslInstalled = $true }
} catch {}

if (-not $wslInstalled) {
    Write-Host "[!] WSL2 not detected. Installing..." -ForegroundColor Yellow
    wsl --install --no-distribution
    Write-Host ""
    Write-Host "[!] WSL2 installed. Restart your computer, then re-run this script." -ForegroundColor Yellow
    return
}
Write-Host "[OK] WSL2 detected" -ForegroundColor Green

# --- Ubuntu distro --------------------------------------------------------
# `wsl --list` emits UTF-16LE; PowerShell string-matching chokes on the null
# bytes, so an existing Ubuntu gets missed and we wrongly try to reinstall
# (ERROR_ALREADY_EXISTS). Strip the nulls before matching.
$distros = ((wsl --list --quiet 2>$null) -join "`n") -replace "`0", ""
if ($distros -notmatch "Ubuntu") {
    Write-Host "[!] Installing Ubuntu in WSL..." -ForegroundColor Yellow
    wsl --install -d Ubuntu --no-launch
    Write-Host "[OK] Ubuntu installed (you may be prompted to create a UNIX user on first launch)" -ForegroundColor Green
} else {
    Write-Host "[OK] Ubuntu already installed" -ForegroundColor Green
}

# --- Docker Engine inside WSL --------------------------------------------
# 2>$null (not 2>&1): under $ErrorActionPreference=Stop, merging WSL's benign
# stderr (e.g. "Processing /etc/fstab with mount -a failed") into a captured
# variable throws a NativeCommandError and kills the install. We only want stdout.
$dockerCheck = wsl -d Ubuntu -- bash -c "command -v docker" 2>$null
if (-not $dockerCheck) {
    Write-Host "[!] Installing Docker Engine in WSL..." -ForegroundColor Yellow
    wsl -d Ubuntu -- bash -c "curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker `$USER"
    Write-Host "[OK] Docker Engine installed in WSL" -ForegroundColor Green
}
Write-Host "[OK] Docker available" -ForegroundColor Green

# Start the Docker daemon if it isn't running.
$dockerInfo = wsl -d Ubuntu -- bash -c "docker info 2>/dev/null" 2>$null
if (-not $dockerInfo) {
    Write-Host "Starting Docker daemon..." -ForegroundColor Yellow
    wsl -d Ubuntu -- bash -c "sudo service docker start"
    Start-Sleep -Seconds 3
}

# --- Project directory ----------------------------------------------------
# If a compose.yaml is already here (cloned repo or previous run), use it.
# Otherwise create .\augmentum and fetch the pull-only compose.yaml + .env.
if (-not (Test-Path "compose.yaml")) {
    $targetDir = if ($env:AUGMENTUM_DIR) { $env:AUGMENTUM_DIR } else { "augmentum" }
    Write-Host "Setting up Augmentum in .\$targetDir ..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    Set-Location $targetDir
    Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/compose.yaml" -OutFile "compose.yaml" -ErrorAction Stop
    # compose.yaml bind-mounts repo config the pull-only install otherwise lacks.
    # searxng needs its settings.yml — without it searxng mounts an empty dir,
    # runs defaults that 403 the json API, and browse/search breaks.
    New-Item -ItemType Directory -Force -Path "config/searxng" | Out-Null
    Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/config/searxng/settings.yml" -OutFile "config/searxng/settings.yml" -ErrorAction Stop
    if (-not (Test-Path ".env")) {
        @"
AUGMENTUM_VARIANT=cpu
# Run the bundled Caddy reverse proxy so the UI is served over HTTPS on :6443
# (self-signed cert -- accept the browser warning once). Plain HTTP stays on
# :6100. HTTPS is required for microphone/camera access from other devices on
# a LAN. Set this empty to disable Caddy and use http://localhost:6100/ui only.
COMPOSE_PROFILES=https
"@ | Out-File -Encoding ascii ".env"
    }
}

# Translate the current Windows path to a WSL /mnt/<drive>/... path.
$currentPath = (Get-Location).Path
if ($currentPath -match "^([A-Za-z]):") {
    $drive = $Matches[1].ToLower()
    $rest = $currentPath.Substring(2) -replace "\\", "/"
    $wslPath = "/mnt/$drive$rest"
} else {
    $wslPath = $currentPath -replace "\\", "/"
}

Write-Host ""
Write-Host "Pulling Augmentum CPU variant from GHCR..." -ForegroundColor Cyan
# Retry the pull: Docker Hub's CDN intermittently throws transient TLS handshake
# timeouts on the small public base images. A retry almost always clears it and
# Docker resumes already-downloaded layers, so a first-time user does not see a
# one-off network flake as an install failure. Kept as ONE line with only bash
# single-quotes and no bash $-vars ($wslPath is substituted by PowerShell): a
# multi-line/apostrophe'd script gets mangled passing through wsl.exe.
$pull = "cd '$wslPath' && for i in 1 2 3 4 5; do docker compose pull && docker compose up -d && exit 0; echo 'pull attempt failed (often a transient Docker Hub CDN hiccup); retrying in 5s...'; sleep 5; done; echo 'Pull kept failing. Almost always an MTU issue on Docker Desktop/WSL2 (TLS handshake timeout on the same layer): open Docker Desktop, Settings, Docker Engine, add mtu 1400 to the JSON, Apply and Restart, then re-run docker compose pull'; exit 1"
wsl -d Ubuntu -- bash -c $pull

Write-Host ""
Write-Host "=== Augmentum is starting ===" -ForegroundColor Green
Write-Host "Open https://localhost:6443 in your browser — it uses a self-signed"
Write-Host "certificate, so accept the warning once. Plain HTTP is also available"
Write-Host "at http://localhost:6100/ui."
Write-Host "The first user to register becomes the admin — pick a username and password."
Write-Host ""
Write-Host "Want the GPU variant or to choose optional services? See the README,"
Write-Host "or run setup.bat from a git clone of the repo."
Write-Host ""
Write-Host "Commands (run inside WSL, from $wslPath):"
Write-Host "  docker compose logs -f    # Watch logs"
Write-Host "  docker compose down       # Stop"
Write-Host "  docker compose up -d      # Start"
Write-Host "  docker compose pull       # Update to the latest image"
