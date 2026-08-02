# Repair a corrupted augmentum.db.
#
# Stops augmentum, runs the repair logic in a transient container that
# mounts the data volume, then restarts augmentum. The repair script
# itself (repair_augmentum_db.py) does all the work; this wrapper just
# orchestrates the container lifecycle so the DB isn't being written
# while it's being rebuilt.
#
# Usage (from repo root):
#   .\scripts\repair_augmentum_db.ps1
#
# Optional env:
#   AUGMENTUM_CONTAINER  default: augmentum-augmentum-1
#   AUGMENTUM_VOLUME     default: augmentum_augmentum_data
#   AUGMENTUM_IMAGE      default: augmentum-augmentum

$ErrorActionPreference = "Stop"

$container = if ($env:AUGMENTUM_CONTAINER) { $env:AUGMENTUM_CONTAINER } else { "augmentum-augmentum-1" }
$volume    = if ($env:AUGMENTUM_VOLUME)    { $env:AUGMENTUM_VOLUME }    else { "augmentum_augmentum_data" }
$image     = if ($env:AUGMENTUM_IMAGE)     { $env:AUGMENTUM_IMAGE }     else { "augmentum-augmentum" }

# Resolve the augmentum image from the container's current config so we
# pick the same one that's running (avoids accidentally using a stale
# tag when the user has rebuilt without retagging "augmentum-augmentum").
$resolvedImage = & docker inspect --format "{{.Config.Image}}" $container 2>$null
if ($LASTEXITCODE -eq 0 -and $resolvedImage) {
    $image = $resolvedImage
}

$scriptHostPath = Join-Path $PSScriptRoot "repair_augmentum_db.py"
if (-not (Test-Path $scriptHostPath)) {
    Write-Error "repair script not found: $scriptHostPath"
    exit 1
}

Write-Host "[wrapper] container: $container"
Write-Host "[wrapper] volume:    $volume"
Write-Host "[wrapper] image:     $image"
Write-Host ""

# 1. Stop the augmentum container so the DB isn't being written.
#    ``docker stop`` sends SIGTERM and waits up to 10s before SIGKILL --
#    enough time for SQLite's checkpoint-on-close to land cleanly.
Write-Host "[wrapper] stopping $container..."
& docker stop $container | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "failed to stop container"
    exit 1
}

# 2. Run the repair inside a transient container that mounts:
#    - the data volume read-write (so the swap can happen)
#    - the repair script read-only (so the container has it)
#
#    The augmentum image ships python3 but not the sqlite3 CLI.
#    Install it transiently so the repair script's preferred path
#    (``sqlite3 .recover`` -- corruption-aware, handles broken B-trees)
#    is available; otherwise it falls back to Python's ``iterdump``
#    which aborts on serious corruption.
Write-Host "[wrapper] running repair..."
$scriptContainerPath = "/tmp/repair_augmentum_db.py"
$repairCmd = "(command -v sqlite3 >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq sqlite3 >/dev/null)) && python3 $scriptContainerPath"
& docker run --rm `
    -v "${volume}:/data" `
    -v "${scriptHostPath}:${scriptContainerPath}:ro" `
    --entrypoint sh `
    --user root `
    $image `
    -c $repairCmd
$repairExit = $LASTEXITCODE

# 3. Always restart augmentum, even on repair failure -- leaving the
#    service down is worse than running on a still-corrupt DB. The user
#    can investigate from logs.
Write-Host ""
Write-Host "[wrapper] starting $container..."
& docker start $container | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "failed to start container after repair"
    exit 1
}

if ($repairExit -ne 0) {
    Write-Host ""
    Write-Warning "repair exited with code $repairExit -- DB may still be corrupt; check repair logs above"
    exit $repairExit
}

Write-Host ""
Write-Host "[wrapper] done."
Write-Host "  - active DB:    /data/augmentum.db (rebuilt)"
Write-Host "  - retired:      /data/augmentum.db.corrupt-<ts>"
Write-Host "  - safety copy:  /data/augmentum.db.backup-<ts>"
Write-Host ""
Write-Host "Inspect via: docker exec $container ls -la /data/"
