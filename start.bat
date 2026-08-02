@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "CONF_FILE=%SCRIPT_DIR%.augmentum.conf"
set "ENV_FILE=%SCRIPT_DIR%.env"

if not exist "%CONF_FILE%" (
  echo.
  echo   No configuration found. Run setup first:
  echo.
  echo     setup.bat     ^(Windows CMD^)
  echo     ./setup.sh    ^(Git Bash^)
  echo.
  exit /b 1
)

:: Read compose file list from config
set /p COMPOSE_FILES=<"%CONF_FILE%"

:: Build -f flags
set "COMPOSE_FLAGS="
for %%f in (%COMPOSE_FILES%) do (
  set "COMPOSE_FLAGS=!COMPOSE_FLAGS! -f %%f"
)

cd /d "%SCRIPT_DIR%"

:: Keep the raw-compose entrypoint in sync: regenerate .env's COMPOSE_FILE (+
:: COMPOSE_PATH_SEPARATOR) from .augmentum.conf so `docker compose up` resolves
:: the SAME overlay set this script does. Non-fatal — this script uses the -f
:: flags built above regardless; the sync only serves the bare-compose path.
where python >nul 2>&1 && (
  python "%SCRIPT_DIR%\scripts\bootstrap\sync_compose_env.py" || echo [start] compose-env sync skipped ^(non-fatal^)
) || (
  where py >nul 2>&1 && py -3 "%SCRIPT_DIR%\scripts\bootstrap\sync_compose_env.py"
)

:: Load .env file if it exists (so explicit operator values win over our
:: auto-detection below).
if exist "%ENV_FILE%" (
  for /f "usebackq tokens=1,* delims==" %%a in ("%ENV_FILE%") do (
    set "%%a=%%b" 2>nul
  )
)

:: Auto-detect host LAN IPs into AUGMENTUM_TLS_EXTRA_SANS when the operator
:: hasn't set it. Without this, Caddy mints a cert that only covers
:: localhost/0.0.0.0 and a phone hitting https://<lan-ip>:6443 sees a
:: cert name-mismatch error the browser may refuse to bypass. PowerShell
:: enumerates IPv4 addresses on all interfaces; we filter to RFC1918 +
:: Tailscale CGNAT (100.64/10) and exclude Docker-bridge gateway IPs
:: (172.17-31.0.1 shape — container-only, never reachable from outside).
if not defined AUGMENTUM_TLS_EXTRA_SANS (
  for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { $_.IPAddress -match '^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.|100\.(6[4-9]|[7-9][0-9]|1[0-1][0-9]|12[0-7])\.)' -and $_.IPAddress -notmatch '^172\.(1[7-9]|2[0-9]|3[01])\.0\.1$' } | ForEach-Object { 'IP:' + $_.IPAddress }) -join ','"`) do (
    set "AUGMENTUM_TLS_EXTRA_SANS=%%i"
  )
  if defined AUGMENTUM_TLS_EXTRA_SANS (
    if not "!AUGMENTUM_TLS_EXTRA_SANS!"=="" (
      echo   TLS SANs auto-detected: !AUGMENTUM_TLS_EXTRA_SANS!
    )
  )
)

:: Auto-detect the node's Tailscale MagicDNS name (<node>.<tailnet>.ts.net) so
:: the app can offer a STABLE tailnet URL and — with Funnel enabled — a durable
:: public guest address, instead of an anonymous cloudflared tunnel whose URL
:: changes each restart. Skipped when Tailscale isn't installed. Operator
:: override (env / .env) wins.
if not defined AUGMENTUM_TAILNET_HOSTNAME (
  where tailscale >nul 2>&1 && (
    for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "try { ((tailscale status --json | ConvertFrom-Json).Self.DNSName).TrimEnd('.') } catch { '' }"`) do (
      set "AUGMENTUM_TAILNET_HOSTNAME=%%i"
    )
    if defined AUGMENTUM_TAILNET_HOSTNAME (
      if not "!AUGMENTUM_TAILNET_HOSTNAME!"=="" (
        echo   Tailscale name: !AUGMENTUM_TAILNET_HOSTNAME! ^(enable Funnel for durable guest access^)
      )
    )
  )
)

:: Build LAN_URLS / TAILSCALE_URLS lists for the banner. PowerShell does
:: the parse + grouping inline so the cmd-side stays readable.
set "LAN_URLS="
set "TAILSCALE_URLS="
if defined AUGMENTUM_TLS_EXTRA_SANS (
  for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$tokens = '!AUGMENTUM_TLS_EXTRA_SANS!' -split ',' | Where-Object { $_ -like 'IP:*' } | ForEach-Object { $_.Substring(3) }; $lan = $tokens | Where-Object { $_ -match '^(10\.|192\.168\.|172\.)' } | ForEach-Object { 'https://' + $_ + ':6443' }; $ts  = $tokens | Where-Object { $_ -match '^100\.(6[4-9]|[7-9][0-9]|1[0-1][0-9]|12[0-7])\.' } | ForEach-Object { 'https://' + $_ + ':6443' }; Write-Output ('LAN=' + ($lan -join ' | ')); Write-Output ('TS=' + ($ts -join ' | '))"`) do (
    set "%%i"
  )
)

:: Export the agent-browser pin so compose.browser.yaml's build arg gets
:: the REAL pinned version, not the Dockerfile default (verified-build-
:: inputs discipline, mirrors start.sh). No-op when the pin is absent.
if exist "%SCRIPT_DIR%AGENT_BROWSER_VERSION" (
  set /p AGENT_BROWSER_VERSION=<"%SCRIPT_DIR%AGENT_BROWSER_VERSION"
)

:: Hidden dispatch: `start.bat __ensure_llama` re-enters this script and
:: runs ONLY the llama-server provisioning label, so the up/-d paths can
:: launch it as a detached background process (cmd can't background an
:: internal label directly). Used below; not a user-facing verb.
if "%~1"=="__ensure_llama" (
  call :ensure_llama_server
  exit /b
)

:: Default action: up (no rebuild unless explicitly requested)
if "%~1"=="" (
  call :ensure_llama_server_async
  echo   Starting Augmentum...
  echo   Config: %COMPOSE_FILES%
  echo   HTTP:  http://localhost:6100
  echo   HTTPS: https://localhost:6443
  if defined LAN if not "!LAN!"=="" echo   HTTPS: !LAN! ^(LAN^)
  if defined TS  if not "!TS!"==""  echo   HTTPS: !TS! ^(Tailscale^)
  echo.
  call :ensure_gs_anchors
  docker compose %COMPOSE_FLAGS% up
  exit /b
)

:: -d shorthand
if "%~1"=="-d" (
  call :ensure_llama_server_async
  echo   Starting Augmentum ^(detached^)...
  echo   Config: %COMPOSE_FILES%
  echo   HTTP:  http://localhost:6100
  echo   HTTPS: https://localhost:6443
  if defined LAN if not "!LAN!"=="" echo   HTTPS: !LAN! ^(LAN^)
  if defined TS  if not "!TS!"==""  echo   HTTPS: !TS! ^(Tailscale^)
  echo.
  call :ensure_gs_anchors
  docker compose %COMPOSE_FLAGS% up -d
  exit /b
)

:: build: rebuild then start
if "%~1"=="build" (
  call :ensure_llama_server
  echo   Building and starting Augmentum...
  echo   Config: %COMPOSE_FILES%
  echo.
  REM Build the workspace profile tags. Multi-stage Dockerfile produces
  REM one tag per profile so workspaces get the right prebake without
  REM re-installing on every create. See spec:
  REM   docs/superpowers/specs/2026-06-02-tooling-profile-system-v2.md
  REM Inside parens, comments MUST use REM not :: — double-colon syntax
  REM still tokenizes brackets/special chars and breaks the enclosing
  REM IF block's parse.
  if exist Dockerfile.workspace (
    echo   Building workspace profile tags ^(standard, power, browser^)...
    echo   Total disk usage ~1.4 GB ^(profiles share layers^).
    docker build -t augmentum-workspace:standard --target standard -f Dockerfile.workspace . >nul 2>&1 || (
      echo   [warning] standard profile build failed — coder mode will use ubuntu:24.04 fallback
    )
    docker build -t augmentum-workspace:power --target power -f Dockerfile.workspace . >nul 2>&1 || (
      echo   [warning] power profile build failed — workspaces with profile=power will fall back to :standard + runtime install
    )
    docker build -t augmentum-workspace:browser --target browser -f Dockerfile.workspace . >nul 2>&1 || (
      echo   [warning] browser profile build failed — workspaces with profile=browser will fall back to :power + runtime install
    )
    REM Pentest profile is opt-in: only built when both pin files exist
    REM AND METASPLOIT_SHA256 has been populated by scripts\upgrade_msf.bat.
    REM The placeholder "UNPINNED" sha would fail the in-Dockerfile sha-check
    REM anyway — skipping with a clear message is friendlier than letting
    REM the build error out mid-run.
    if exist METASPLOIT_VERSION if exist METASPLOIT_SHA256 (
      set /p MSF_VERSION=<METASPLOIT_VERSION
      set /p MSF_SHA=<METASPLOIT_SHA256
      if not "!MSF_SHA!"=="" if not "!MSF_SHA!"=="UNPINNED" (
        echo   Building pentest profile ^(Metasploit !MSF_VERSION!, ~4 GB^)...
        docker build -t augmentum-workspace:pentest --target pentest ^
          --build-arg METASPLOIT_VERSION=!MSF_VERSION! ^
          --build-arg METASPLOIT_SHA256=!MSF_SHA! ^
          -f Dockerfile.workspace . >nul 2>&1 || (
          echo   [warning] pentest profile build failed — workspaces with profile=pentest will fall back to :power + runtime install
        )
      ) else (
        echo   [info] pentest profile skipped — METASPLOIT_SHA256 is unset/placeholder.
        echo            Run scripts\upgrade_msf.bat to populate the pin, then rerun start.bat build.
      )
    )
    REM Backward compat: keep the v1 unversioned tag pointing at browser
    REM ^(the default profile^) so callers that hardcoded "augmentum-workspace"
    REM keep working without code change.
    docker tag augmentum-workspace:browser augmentum-workspace:latest >nul 2>&1 || (echo   [info] could not tag :latest)
    docker tag augmentum-workspace:browser augmentum-workspace >nul 2>&1 || (echo   [info] could not tag generic alias)
  )
  REM Build the game-stream images when game streaming is enabled in .augmentum.conf
  findstr /c:"compose.game-stream.yaml" "%CONF_FILE%" >nul 2>&1
  if not errorlevel 1 call :build_game_stream
  REM Re-anchor AFTER the builds: a rebuilt image has a new ID, and an
  REM anchor still pinning the OLD one leaves the new image unprotected.
  call :ensure_gs_anchors
  docker compose %COMPOSE_FLAGS% up --build
  exit /b
)

:: Pass through any other subcommand
docker compose %COMPOSE_FLAGS% %*
exit /b

:: ============================================================
:: Subroutines
:: ============================================================

:ensure_gs_anchors
:: Keep the game-stream images referenced by a running container so a
:: routine `docker image prune -a` / Docker Desktop "Clean up" can't
:: sweep them as unused -- they're spawned on demand, so between
:: sessions nothing holds them and they look like garbage. Also warns
:: when an image isn't built, so that surfaces at startup instead of
:: as a per-title 404 at launch. Non-fatal by construction: the script
:: always exits 0 and this must never block boot.
where python >nul 2>&1 && (
  python "%SCRIPT_DIR%\scripts\bootstrap\ensure_game_stream_anchors.py" 2>nul
) || (
  where py >nul 2>&1 && py -3 "%SCRIPT_DIR%\scripts\bootstrap\ensure_game_stream_anchors.py" 2>nul
)
goto :eof

:build_game_stream
:: Build the AGSP container images. Runs only when
:: compose.game-stream.yaml is present in .augmentum.conf (caller's
:: precondition). On failure of either build, warns the user but does
:: not abort the outer ``start.bat build`` -- streaming will simply
:: be unavailable until they fix the build.
echo   Building game-stream base image...
docker build -t augmentum-game-stream-base -f services\game-stream\Dockerfile.base .
if errorlevel 1 (
  echo   [warning] Game-stream base build failed -- streaming will be unavailable
  goto :eof
)
echo   Building game-stream Luanti image...
docker build -t augmentum-game-stream-luanti -f services\game-stream\Dockerfile.luanti .
if errorlevel 1 (
  echo   [warning] Game-stream Luanti build failed -- streaming will be unavailable
)
:: Emulator-streamed image (Dolphin + PCSX2) is in compose profile
:: build-only, so `docker compose up --build` skips it. Build it
:: explicitly so streamed GameCube/Wii/PS2 titles work out of the box.
echo   Building game-stream emulator-streamed image (Dolphin + PCSX2)...
docker build -t augmentum-game-stream-emulator-streamed -f services\game-stream\Dockerfile.emulator-streamed .
if errorlevel 1 (
  echo   [warning] Game-stream emulator-streamed build failed -- GameCube/Wii/PS2 will be unavailable
)
:: Browser-stream image. Also build-only in compose, so `up --build`
:: skips it too -- without this line cast-to-TV of web surfaces has no
:: image on any fresh install and fails with a 404 at cast time.
echo   Building stream-browser image (cast-to-TV of web surfaces)...
docker build -t augmentum-stream-browser -f services\game-stream\Dockerfile.browser .
if errorlevel 1 (
  echo   [warning] stream-browser build failed -- casting web surfaces will be unavailable
)
goto :eof

:ensure_llama_server_async
:: Background wrapper (2026-07-02 boot-latency work). The RUNNING stack
:: never needs the llama-server builder image — the binary is baked into
:: the app image; it only feeds the next `start.bat build`. When the
:: image exists this is a ~100ms no-op; when missing (post-prune, fresh
:: checkout) the GHCR pull / 30-50 min CUDA compile used to block the
:: entire startup. Now it runs as a detached process via the
:: __ensure_llama self-dispatch at the top of this script; progress
:: lands in %TEMP%\augmentum-llama-ensure.log. The `build` path still
:: calls :ensure_llama_server inline — that path genuinely needs the
:: image before building the app image.
docker image inspect augmentum-llama-server >nul 2>&1 && goto :eof
echo   llama-server builder image missing -- provisioning in background
echo   ^(log: %TEMP%\augmentum-llama-ensure.log; only needed by 'start.bat build'^)
start "" /b cmd /c ""%~f0" __ensure_llama > "%TEMP%\augmentum-llama-ensure.log" 2>&1"
goto :eof

:ensure_llama_server
:: Provision augmentum-llama-server when a local build of augmentum will
:: fire (compose.dev.yaml has a build: directive that COPYs the binary
:: out of FROM augmentum-llama-server:latest). Production runs that pull
:: the prebuilt augmentum image from GHCR don't need this.
:: Strategy: prefer ~30s GHCR pull, fall back to ~30-50 min local CUDA
:: compile only if pull fails or repo lacks a published binary.
docker image inspect augmentum-llama-server >nul 2>&1 && goto :eof
findstr /c:"compose.dev.yaml" "%CONF_FILE%" >nul 2>&1 || goto :eof
if not exist "%SCRIPT_DIR%LLAMA_SERVER_VERSION" goto :eof
set /p LLAMA_VER=<"%SCRIPT_DIR%LLAMA_SERVER_VERSION"
echo   Fetching llama-server !LLAMA_VER! from GHCR...
docker pull ghcr.io/augmentumhq/augmentum-llama-server:!LLAMA_VER! >nul 2>&1
if !errorlevel! equ 0 (
  docker tag ghcr.io/augmentumhq/augmentum-llama-server:!LLAMA_VER! augmentum-llama-server:latest >nul 2>&1
  echo     Pulled augmentum-llama-server:latest from GHCR.
  goto :eof
)
if not exist "%SCRIPT_DIR%Dockerfile.llama-server" (
  echo   [warning] llama-server image unavailable -- built-in engine will be unavailable
  goto :eof
)
echo     GHCR pull failed -- falling back to local CUDA compile ^(~30-50 min^)...
docker build -t augmentum-llama-server --build-arg LLAMA_CPP_VERSION=!LLAMA_VER! --progress=plain -f Dockerfile.llama-server . || (
  echo   [warning] llama-server build failed -- built-in engine will be unavailable
)
goto :eof
