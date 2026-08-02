# ────────────────────────────────────────────────────────────────────
# install.ps1 — set up `claude-aug` (Claude Code → local Augmentum)
#
#   .\scripts\claude-aug\install.ps1
#   .\scripts\claude-aug\install.ps1 -ApiKey sk-aug-... -BaseUrl http://localhost:6100
#
# Idempotent: re-run any time to update. Existing claude.env values are
# kept as defaults so re-installing never wipes your config.
# ────────────────────────────────────────────────────────────────────
param(
    [string]$ApiKey,
    [string]$BaseUrl,
    [string]$MainModel,
    [string]$SmallModel
)
$ErrorActionPreference = 'Stop'
$src    = $PSScriptRoot
$augDir = Join-Path $HOME '.augmentum'
$cfgDir = Join-Path $augDir 'claude-config'
$envFile = Join-Path $augDir 'claude.env'

# ── Defaults: pull from existing claude.env if present ───────────────
function Get-EnvFileVal([string]$name, [string]$default) {
    if (Test-Path $envFile) {
        $m = Select-String -Path $envFile -Pattern "^export $name=(.*)$" | Select-Object -First 1
        if ($m) { return $m.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'") }
    }
    return $default
}
if (-not $ApiKey)     { $ApiKey     = Get-EnvFileVal 'AUGMENTUM_API_KEY' '' }
if (-not $BaseUrl)    { $BaseUrl    = Get-EnvFileVal 'ANTHROPIC_BASE_URL' 'http://localhost:6100' }
if (-not $MainModel)  { $MainModel  = Get-EnvFileVal 'ANTHROPIC_MODEL' 'deepseek-v4-pro' }
if (-not $SmallModel) { $SmallModel = Get-EnvFileVal 'ANTHROPIC_SMALL_FAST_MODEL' 'deepseek-v4-flash' }

if (-not $ApiKey) {
    $ApiKey = Read-Host 'Augmentum API key (Settings -> API Keys)'
    if (-not $ApiKey) { Write-Error 'API key is required'; exit 1 }
}

New-Item -ItemType Directory -Force -Path $augDir, $cfgDir | Out-Null

# ── 1. claude.env from template ──────────────────────────────────────
$envContent = (Get-Content (Join-Path $src 'claude.env.template') -Raw) `
    -replace '\{\{API_KEY\}\}', $ApiKey `
    -replace '\{\{BASE_URL\}\}', $BaseUrl `
    -replace '\{\{MAIN_MODEL\}\}', $MainModel `
    -replace '\{\{SMALL_MODEL\}\}', $SmallModel
Set-Content -Path $envFile -Value $envContent -NoNewline

# ── 2. Copy portable files ───────────────────────────────────────────
Copy-Item (Join-Path $src 'claude-aug.sh')      $augDir -Force
Copy-Item (Join-Path $src 'claude-aug.ps1')     $augDir -Force
Copy-Item (Join-Path $src 'atp-mcp-bridge.py')  $augDir -Force
Copy-Item (Join-Path $src 'bridge-hooks.py')    $augDir -Force
Copy-Item (Join-Path $src 'doctor.py')          $augDir -Force
Copy-Item (Join-Path $src 'statusline.ps1')     $cfgDir -Force
Copy-Item (Join-Path $src 'CLAUDE.md')          $cfgDir -Force
Copy-Item (Join-Path $src 'skills')             $cfgDir -Recurse -Force

# ── 3. Generate claude-config (paths + key hash are user-specific) ──
$fwdHome   = $HOME -replace '\\', '/'
$bridge    = "$fwdHome/.augmentum/atp-mcp-bridge.py"
$hookScript = "$fwdHome/.augmentum/bridge-hooks.py"
$statusline = "$fwdHome/.augmentum/claude-config/statusline.ps1"
# Claude Code identifies an approved custom API key by its last 20 chars
$keyHash = if ($ApiKey.Length -ge 20) { $ApiKey.Substring($ApiKey.Length - 20) } else { $ApiKey }

@{
    hasCompletedOnboarding = $true
    customApiKeyResponses  = @{ approved = @($keyHash); rejected = @() }
    mcpServers = @{
        atp = @{ type = 'stdio'; command = 'python'; args = @($bridge) }
    }
} | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $cfgDir '.claude.json')

@{
    statusLine = @{
        type    = 'command'
        command = "powershell -NoProfile -ExecutionPolicy Bypass -File $statusline"
    }
    permissions = @{
        deny  = @('WebSearch', 'WebFetch')   # dead Anthropic server-tools on a local backend
        allow = @('mcp__atp')                # Augmentum Tool Protocol bridge
    }
    # Agent-bridge hooks: register presence + pick up tasks the user queued for
    # this machine (SessionStart/Stop), and route tool-permission gates to the
    # user's phone (PreToolUse/Notification). See bridge-hooks.py.
    hooks = @{
        SessionStart = @( @{ matcher = ''; hooks = @( @{ type = 'command'; command = "python $hookScript --hook SessionStart" } ) } )
        Stop         = @( @{ matcher = ''; hooks = @( @{ type = 'command'; command = "python $hookScript --hook Stop" } ) } )
        PreToolUse   = @( @{ matcher = 'Bash|Edit|Write|NotebookEdit|Agent|Workflow|Task'; hooks = @( @{ type = 'command'; command = "python $hookScript --hook PreToolUse" } ) } )
        Notification = @( @{ matcher = 'permission'; hooks = @( @{ type = 'command'; command = "python $hookScript --hook PreToolUse" } ) } )
    }
} | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $cfgDir 'settings.json')

# ── 4. Detect the Augmentum container (for wrapper auto-start) ──────
$container = ''
try {
    $container = (docker ps -a --format '{{.Names}}' 2>$null |
        Where-Object { $_ -match '^augmentum.*augmentum' } | Select-Object -First 1)
} catch {}
if ($container -and $container -ne 'augmentum-augmentum-1') {
    $wrapper = Join-Path $augDir 'claude-aug.ps1'
    (Get-Content $wrapper -Raw) -replace 'augmentum-augmentum-1', $container |
        Set-Content $wrapper -NoNewline
    Write-Host "Wrapper auto-start wired to container: $container"
}

# ── 5. Hook into the PowerShell profile ─────────────────────────────
$hook = ". `"`$HOME\.augmentum\claude-aug.ps1`""
if (-not (Test-Path $PROFILE)) { New-Item -ItemType File -Force -Path $PROFILE | Out-Null }
if (-not (Select-String -Path $PROFILE -Pattern 'claude-aug\.ps1' -Quiet)) {
    Add-Content $PROFILE "`n# Claude Code via local Augmentum`n$hook"
    Write-Host "Added claude-aug to $PROFILE"
}

# ── 6. Remind about server-side aliases ─────────────────────────────
Write-Host ''
Write-Host 'Installed. Open a NEW terminal and run: claude-aug' -ForegroundColor Green
Write-Host ''
Write-Host 'One server-side step (needed once): ensure these are in Augmentum''s .env'
Write-Host 'so the container maps Claude Code''s hardcoded claude-* model IDs:'
Write-Host "  AUGMENTUM_ANTHROPIC_ALIAS_HAIKU=$SmallModel"
Write-Host "  AUGMENTUM_ANTHROPIC_ALIAS_SONNET=$SmallModel"
Write-Host "  AUGMENTUM_ANTHROPIC_ALIAS_OPUS=$MainModel"
Write-Host "  AUGMENTUM_ANTHROPIC_ALIAS_DEFAULT=$MainModel"
