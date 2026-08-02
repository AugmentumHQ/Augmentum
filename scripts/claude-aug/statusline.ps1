# Statusline for claude-aug sessions: makes it unmistakable you're on Augmentum.
$input_json = [Console]::In.ReadToEnd()
try { $data = $input_json | ConvertFrom-Json } catch { $data = $null }
$model = if ($data -and $data.model.display_name) { $data.model.display_name } else { $env:ANTHROPIC_MODEL }
$server = if ($env:ANTHROPIC_BASE_URL) { $env:ANTHROPIC_BASE_URL -replace '^https?://', '' } else { '?' }
Write-Output "AUG | $model @ $server"
