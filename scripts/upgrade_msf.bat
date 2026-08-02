@echo off
setlocal EnableDelayedExpansion

REM Upgrade the pinned Metasploit Framework .deb used by the pentest
REM workspace profile (augmentum-workspace:pentest).
REM
REM Usage:
REM   scripts\upgrade_msf.bat              use currently-pinned version, recompute sha
REM   scripts\upgrade_msf.bat 6.4.42       pin to specific version
REM
REM Requires: curl (Windows 10+ ships it) and certutil (built in) for sha256.

set REPO_ROOT=%~dp0..
set VERSION_FILE=%REPO_ROOT%\METASPLOIT_VERSION
set SHA_FILE=%REPO_ROOT%\METASPLOIT_SHA256

set TARGET=%~1

if "%TARGET%"=="" (
    set /p TARGET=<"%VERSION_FILE%"
    echo Using currently-pinned version: !TARGET!
)

REM Validate semver shape loosely.
echo %TARGET% | findstr /r "^[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*$" >nul
if errorlevel 1 (
    echo ERROR: version '%TARGET%' does not look like a metasploit-framework release ^(X.Y.Z^)
    exit /b 1
)

set DEB_NAME=metasploit-framework_%TARGET%-1rapid7-1_amd64.deb
set DEB_URL=https://apt.metasploit.com/pool/main/m/metasploit-framework/%DEB_NAME%
set TMP_DEB=%TEMP%\augmentum_msf_%RANDOM%.deb

echo.
echo Fetching %DEB_NAME%...
curl -fsSL -o "%TMP_DEB%" "%DEB_URL%"
if errorlevel 1 (
    echo ERROR: download failed. Confirm the version exists at:
    echo   %DEB_URL%
    if exist "%TMP_DEB%" del /q "%TMP_DEB%"
    exit /b 1
)

echo Computing sha256...
for /f "skip=1 tokens=1" %%i in ('certutil -hashfile "%TMP_DEB%" SHA256 ^| findstr /v ":"') do (
    if not defined SHA set SHA=%%i
)
set SHA=%SHA: =%

echo   sha256: %SHA%

REM Persist pins atomically (write both or neither).
^> "%VERSION_FILE%.tmp" echo %TARGET%
^> "%SHA_FILE%.tmp" echo %SHA%
move /Y "%VERSION_FILE%.tmp" "%VERSION_FILE%" >nul
move /Y "%SHA_FILE%.tmp" "%SHA_FILE%" >nul

del /q "%TMP_DEB%"

echo.
echo [OK] Pinned METASPLOIT_VERSION=%TARGET%
echo [OK] Pinned METASPLOIT_SHA256=%SHA%
echo.
echo Next: rebuild the pentest image
echo   start.bat build               (rebuilds all profile tags)
echo   -- or --
echo   docker build -t augmentum-workspace:pentest --target pentest ^^
echo     --build-arg METASPLOIT_VERSION=%TARGET% ^^
echo     --build-arg METASPLOIT_SHA256=%SHA% ^^
echo     -f Dockerfile.workspace .
