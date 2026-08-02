@echo off
setlocal EnableDelayedExpansion

REM Upgrade the bundled llama-server binary (Engine v2's inference core).
REM
REM Usage:
REM   scripts\upgrade_llama_server.bat              use current pin
REM   scripts\upgrade_llama_server.bat b8839        pin to specific tag
REM   scripts\upgrade_llama_server.bat --latest     fetch latest released tag

set REPO_ROOT=%~dp0..
set VERSION_FILE=%REPO_ROOT%\LLAMA_SERVER_VERSION
set DOCKERFILE=%REPO_ROOT%\Dockerfile.llama-server

set TARGET=%~1

if "%TARGET%"=="--latest" (
    echo Fetching latest llama.cpp tag from GitHub...
    for /f "tokens=*" %%i in ('curl -fsSL https://api.github.com/repos/ggml-org/llama.cpp/releases/latest ^| findstr "tag_name"') do set LINE=%%i
    for /f "tokens=2 delims=:," %%i in ("!LINE!") do set TARGET=%%i
    set TARGET=!TARGET:"=!
    set TARGET=!TARGET: =!
    echo Latest release tag: !TARGET!
) else if "%TARGET%"=="" (
    set /p TARGET=<"%VERSION_FILE%"
    echo Using currently-pinned version: !TARGET!
)

REM Validate tag shape loosely (bNNNN).
echo %TARGET% | findstr /r "^b[0-9][0-9]*$" >nul
if errorlevel 1 (
    echo ERROR: tag '%TARGET%' does not look like a llama.cpp release tag ^(bNNNN^)
    exit /b 1
)

echo %TARGET% > "%VERSION_FILE%"

echo.
echo Building augmentum-llama-server:%TARGET% from Dockerfile.llama-server...
echo   ^(this is the long step — expect ~10-25 min on first build^)
echo.

docker build ^
    --build-arg LLAMA_CPP_VERSION=%TARGET% ^
    -t augmentum-llama-server:%TARGET% ^
    -t augmentum-llama-server:latest ^
    -f "%DOCKERFILE%" ^
    "%REPO_ROOT%"
if errorlevel 1 exit /b 1

echo.
echo [DONE] augmentum-llama-server:%TARGET% built and tagged :latest

rem -- Behavior-defaults diff guard -------------------------------------
rem Upstream changes flag DEFAULTS between releases (real incident: b9181
rem flipped cache_idle_slots to default-ON, silently killing KV slot
rem routing). Snapshot --help per version; diff manually or via the .sh
rem sibling, which prints default changes automatically.
if not exist "%REPO_ROOT%\docs\llama-server-defaults" mkdir "%REPO_ROOT%\docs\llama-server-defaults"
docker run --rm --entrypoint llama-server augmentum-llama-server:%TARGET% --help > "%REPO_ROOT%\docs\llama-server-defaults\help-%TARGET%.txt" 2>&1
echo Saved flag/default snapshot: docs\llama-server-defaults\help-%TARGET%.txt
echo Compare against the previous help-b*.txt for changed 'default:' lines --
echo a flipped default is a behavior change Augmentum inherits with no code diff.

echo.
echo Next:  docker compose build augmentum ^&^& docker compose up -d augmentum
echo Verify: docker exec augmentum-augmentum-1 llama-server --version
