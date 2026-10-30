# Production wall-pad node on Windows (Surface). Reads .env.wallpad, no preview.
#
#   .\scripts\wallpad-windows.ps1
#   .\scripts\wallpad-windows.ps1 -EnvFile .env.other -Venv .venv-wallpad
#
# Unlike dev-windows.ps1 this does NOT install anything on every run -- it fails
# fast if the venv is missing, so a kiosk restart can't stall on a pip download.
param(
    [string]$EnvFile = ".env.wallpad",
    [string]$Venv = ".venv-dev"
)

$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$python = Join-Path $root "$Venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "No venv at $Venv -- create it once with: py -3.10 -m venv $Venv; .\$Venv\Scripts\python.exe -m pip install -r requirements.txt; .\$Venv\Scripts\python.exe -m pip install --no-deps -e ."
    exit 1
}
if (-not (Test-Path $EnvFile)) {
    Write-Error "No $EnvFile -- copy .env.example and fill in KK_HA_TOKEN."
    exit 1
}

# KEY=VALUE lines only; '#' comments and blanks ignored. Values are taken
# literally (no quote stripping) so tokens with '=' survive intact.
$loaded = 0
foreach ($line in Get-Content $EnvFile) {
    if ($line -match '^\s*(#|$)') { continue }
    $pair = $line -split '=', 2
    if ($pair.Count -ne 2) { continue }
    $name = $pair[0].Trim()
    if (-not $name) { continue }
    Set-Item -Path "Env:$name" -Value $pair[1].Trim()
    $loaded++
}
Write-Host "loaded $loaded vars from $EnvFile"

if (-not $env:KK_HA_TOKEN) {
    Write-Warning "KK_HA_TOKEN is empty -- the node will start in dry-run mode."
}
$port = if ($env:KK_PORT) { $env:KK_PORT } else { "8420" }
Write-Host "Web UI: http://127.0.0.1:$port  (Ctrl+C to stop)"
& $python -m kinectknob
