# Development mode on Windows: run the full pipeline with your webcam,
# no Kinect and no Home Assistant needed (dry-run volume simulation).
#
#   .\scripts\dev-windows.ps1              # webcam + web UI at http://localhost:8420
#   .\scripts\dev-windows.ps1 -Preview     # + native preview window (press q to quit)
#
# To also drive your real Home Assistant from your PC:
#   $env:KK_HA_URL = "http://192.168.1.10:8123"
#   $env:KK_HA_TOKEN = "<long-lived token>"
#   $env:KK_VOLUME_ENTITY = "media_player.bose_soundbar_700"
#   $env:KK_MEDIA_ENTITY = "media_player.spotify_yash"
param([switch]$Preview)

$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

# mediapipe ships wheels for Python 3.9-3.12 only.
$py = $null
foreach ($v in @("3.12", "3.11", "3.10")) {
    try { & py "-$v" --version *> $null; if ($LASTEXITCODE -eq 0) { $py = $v; break } } catch {}
}
if (-not $py) {
    Write-Error "Python 3.10-3.12 required (mediapipe has no 3.13 wheels yet). Install from python.org, e.g. 3.12."
    exit 1
}

if (-not (Test-Path ".venv-dev")) {
    Write-Host "Creating dev venv with Python $py..."
    & py "-$py" -m venv .venv-dev
}
& .\.venv-dev\Scripts\python.exe -m pip install --quiet --upgrade pip
& .\.venv-dev\Scripts\python.exe -m pip install --quiet -r requirements.txt
# Editable install so `python -m kinectknob` resolves from src/.
& .\.venv-dev\Scripts\python.exe -m pip install --quiet --no-deps -e .

$args = @("-m", "kinectknob", "--backend", "webcam")
if ($Preview) { $args += "--preview" }
Write-Host "Web UI: http://localhost:8420  (Ctrl+C to stop)"
& .\.venv-dev\Scripts\python.exe @args
