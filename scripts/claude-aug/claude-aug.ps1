# ────────────────────────────────────────────────────────────────────
# claude-aug — Claude Code routed through your LOCAL Augmentum proxy.
#
#   claude-aug                              # default models from claude.env
#   claude-aug models                       # list available Augmentum models
#   claude-aug --model qwen3.6-35b          # one-off main model switch
#   claude-aug --small deepseek-v4-flash    # one-off subagent model
#   claude-aug -p "summarize this repo"     # any claude args pass through
#
# Normal `claude` is untouched — still hits Anthropic with your sub.
# Configuration: ~/.augmentum/claude.env
#
# To install: add this file, then add to your $PROFILE:
#   . "$HOME\.augmentum\claude-aug.ps1"
# ────────────────────────────────────────────────────────────────────

function claude-aug {
    $ErrorActionPreference = 'Stop'
    
    # ── Helper: get env var with fallback (PS 5.1 safe) ──────────
    function Get-EnvOrDefault {
        param([string]$Name, $Default = '')
        $v = [Environment]::GetEnvironmentVariable($Name, 'Process')
        if ($v) { return $v }
        return $Default
    }

    # ── Snapshot session env before claude.env pollutes it ───────
    $scopedVars = @('ANTHROPIC_BASE_URL','ANTHROPIC_API_KEY','ANTHROPIC_AUTH_TOKEN',
                    'ANTHROPIC_MODEL','ANTHROPIC_SMALL_FAST_MODEL','ANTHROPIC_CUSTOM_HEADERS',
                    'CLAUDE_CONFIG_DIR','CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC','API_TIMEOUT_MS',
                    'CLAUDE_CODE_SUBAGENT_MODEL','MCP_TOOL_TIMEOUT')
    $saved = @{}
    foreach ($v in $scopedVars) { $saved[$v] = [Environment]::GetEnvironmentVariable($v, 'Process') }

    # ── Load config from claude.env ──────────────────────────────
    $augDir = Join-Path $HOME '.augmentum'
    $envFile = Join-Path $augDir 'claude.env'
    if (-not (Test-Path $envFile)) {
        Write-Error "Config not found: $envFile"
        return 1
    }
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -match '^\s*export\s+([A-Z_]+)=(.*)') {
            $name = $matches[1]
            $value = $matches[2].Trim().Trim('"').Trim("'")
            # Expand ${VAR} references with process env lookup
            $value = [regex]::Replace($value, '\$\{([^}]+)\}', {
                param($m)
                $v = [Environment]::GetEnvironmentVariable($m.Groups[1].Value, 'Process')
                if ($v) { $v } else { '' }
            })
            [Environment]::SetEnvironmentVariable($name, $value, 'Process')
        }
    }

    # ── Defaults ─────────────────────────────────────────────────
    $baseUrl    = Get-EnvOrDefault 'ANTHROPIC_BASE_URL' 'http://localhost:6100'
    $apiKey     = Get-EnvOrDefault 'ANTHROPIC_API_KEY' (Get-EnvOrDefault 'AUGMENTUM_API_KEY')
    $model      = Get-EnvOrDefault 'ANTHROPIC_MODEL' 'deepseek-v4-flash'
    $smallModel = Get-EnvOrDefault 'ANTHROPIC_SMALL_FAST_MODEL' 'deepseek-v4-flash'
    $aliasHaiku  = Get-EnvOrDefault 'AUGMENTUM_ANTHROPIC_ALIAS_HAIKU' $smallModel
    $aliasSonnet = Get-EnvOrDefault 'AUGMENTUM_ANTHROPIC_ALIAS_SONNET' $model
    $aliasOpus   = Get-EnvOrDefault 'AUGMENTUM_ANTHROPIC_ALIAS_OPUS' $model

    # ── Parse args ───────────────────────────────────────────────
    $claudeArgs = @()
    $i = 0
    while ($i -lt $args.Count) {
        switch ($args[$i]) {
            'models' {
                Write-Host "Fetching models from $baseUrl ..."
                try {
                    $wc = New-Object System.Net.WebClient
                    $wc.Headers.Add('x-api-key', $apiKey)
                    $wc.Headers.Add('anthropic-version', '2023-06-01')
                    $json = $wc.DownloadString("$baseUrl/v1/models")
                    $wc.Dispose()
                    $data = $json | ConvertFrom-Json
                    $names = ($data.data | Where-Object { $_.id -notmatch '^[an]/' } | ForEach-Object { $_.id } | Sort-Object -Unique)
                    $names | ForEach-Object { Write-Host "  $_" }
                    Write-Host ""
                    Write-Host "$($names.Count) models available"
                } catch {
                    Write-Host "ERROR: Could not reach Augmentum at $baseUrl" -ForegroundColor Red
                }
                return
            }
            'doctor' {
                & python (Join-Path $augDir 'doctor.py') @($args | Select-Object -Skip ($i + 1))
                return $LASTEXITCODE
            }
            '--profile' {
                $i++
                if ($i -ge $args.Count) { Write-Error "Missing profile name after --profile"; return 1 }
                $profileVal = Get-EnvOrDefault ("AUGMENTUM_PROFILE_" + $args[$i].ToUpper())
                if (-not $profileVal) {
                    Write-Error "Unknown profile '$($args[$i])' - define AUGMENTUM_PROFILE_$($args[$i].ToUpper()) in claude.env"
                    return 1
                }
                $parts = $profileVal -split '\s+'
                $model = $parts[0]
                $smallModel = if ($parts.Count -gt 1) { $parts[1] } else { $parts[0] }
            }
            '--model' {
                $i++
                if ($i -ge $args.Count) { Write-Error "Missing model name after --model"; return 1 }
                $model = $args[$i]
            }
            '--small' {
                $i++
                if ($i -ge $args.Count) { Write-Error "Missing model name after --small"; return 1 }
                $smallModel = $args[$i]
            }
            { $_ -in @('--help', '-h', '-?') } {
                Write-Host "Usage: claude-aug [--profile NAME] [--model MODEL] [--small MODEL] [models|--help] [claude args...]"
                Write-Host ""
                Write-Host "  models          List available Augmentum models"
                Write-Host "  --profile NAME  Use a model profile (deep/fast/mixed, from claude.env)"
                Write-Host "  --model NAME    Override main model for this session"
                Write-Host "  --small NAME    Override subagent model for this session"
                Write-Host "  --help          This message"
                Write-Host ""
                Write-Host "Default config:  ~\.augmentum\claude.env"
                Write-Host "Normal 'claude' is untouched - still hits Anthropic directly."
                return
            }
            default {
                $claudeArgs += $args[$i]
            }
        }
        $i++
    }

    # ── Health check (auto-starts the container once if down) ────
    function Test-Augmentum {
        try {
            $wc = New-Object System.Net.WebClient
            $wc.Headers.Add('x-api-key', $apiKey)
            $wc.Headers.Add('anthropic-version', '2023-06-01')
            $null = $wc.DownloadString("$baseUrl/v1/models")
            $wc.Dispose()
            return $null
        } catch {
            return $_.Exception.Message
        }
    }

    $healthErr = Test-Augmentum
    if ($healthErr -and $healthErr -notmatch '\((\d+)\)') {
        Write-Host "Augmentum not reachable — trying to start augmentum-augmentum-1 ..." -ForegroundColor Yellow
        docker start augmentum-augmentum-1 2>$null | Out-Null
        foreach ($wait in 3, 5, 8) {
            Start-Sleep -Seconds $wait
            $healthErr = Test-Augmentum
            if (-not $healthErr) { break }
        }
    }
    if ($healthErr) {
        if ($healthErr -match '\((\d+)\)') {
            Write-Host "Augmentum returned HTTP $($matches[1]) at $baseUrl" -ForegroundColor Red
            Write-Host "Check your API key in ~\.augmentum\claude.env"
        } else {
            Write-Host "Cannot reach Augmentum at $baseUrl" -ForegroundColor Red
            Write-Host "Tried auto-starting the container; check: docker ps -a"
        }
        return 1
    }

    # ── Status banner ────────────────────────────────────────────
    Write-Host "+-- Augmentum ----------------------------------------------------+"
    Write-Host ("|  {0,-12} {1,-50} |" -f 'Server:', $baseUrl)
    Write-Host ("|  {0,-12} {1,-50} |" -f 'Main:', $model)
    Write-Host ("|  {0,-12} {1,-50} |" -f 'Subagents:', $smallModel)
    Write-Host ("|  {0,-12} {1,-50} |" -f 'Aliases:', "haiku->$aliasHaiku  sonnet->$aliasSonnet  opus->$aliasOpus")
    Write-Host "+-----------------------------------------------------------------+"

    # ── Route to Augmentum (scoped: restore session env afterwards) ──
    try {
        $env:ANTHROPIC_BASE_URL        = $baseUrl
        $env:ANTHROPIC_API_KEY         = $apiKey
        # (ANTHROPIC_AUTH_TOKEN no longer set: the pre-approved key in
        # claude-config handles the OAuth prompt, and setting both triggers
        # a dual-auth warning in Claude Code.)
        $env:ANTHROPIC_MODEL           = $model
        $env:ANTHROPIC_SMALL_FAST_MODEL = $smallModel
        $env:CLAUDE_CODE_SUBAGENT_MODEL = $smallModel  # actual subagent model control
        # Harness + project identity: project slug scopes server-side memory
        # (proxy/harness.py) so conventions/facts don't bleed across projects.
        $projSlug = Split-Path -Leaf (Get-Location)
        $env:ANTHROPIC_CUSTOM_HEADERS  = "X-Augmentum-Harness: claude_code`nX-Augmentum-Project: $projSlug"

        # Fully isolated config: own ~/.claude.json, settings, sessions
        $env:CLAUDE_CONFIG_DIR = Join-Path $augDir 'claude-config'
        # No stray Anthropic calls (telemetry, update checks) while on local backend
        $env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = '1'
        # Local models are slower than the API — allow long generations
        $env:API_TIMEOUT_MS = '600000'
        # ATP research/browser calls can run 1-2 min; don't let CC's MCP
        # timeout kill them mid-flight
        $env:MCP_TOOL_TIMEOUT = '300000'

        & claude @claudeArgs
    } finally {
        foreach ($v in $scopedVars) {
            [Environment]::SetEnvironmentVariable($v, $saved[$v], 'Process')
        }
    }
}
